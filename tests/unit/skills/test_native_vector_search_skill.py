"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for NativeVectorSearchSkill
"""

import pytest
import os
import logging
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from signalwire_agents.core.function_result import SwaigFunctionResult


# ---------------------------------------------------------------------------
# Helper: create a NativeVectorSearchSkill instance with a mocked agent,
# bypassing any heavy imports that the real __init__ chain may trigger.
# ---------------------------------------------------------------------------

def _make_skill(params=None):
    """Instantiate NativeVectorSearchSkill with a mocked agent."""
    from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill

    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    mock_agent.prompt_add_section = Mock()
    mock_agent.prompt_add_to_section = Mock()
    mock_agent.prompt_has_section = Mock(return_value=False)

    skill = NativeVectorSearchSkill(agent=mock_agent, params=params or {})
    return skill


# ===========================================================================
# Class attributes and parameter schema
# ===========================================================================

class TestSkillClassAttributes:
    """Verify class-level constants on NativeVectorSearchSkill."""

    def test_skill_name(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        assert NativeVectorSearchSkill.SKILL_NAME == "native_vector_search"

    def test_skill_description(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        assert "vector" in NativeVectorSearchSkill.SKILL_DESCRIPTION.lower()

    def test_skill_version(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        assert NativeVectorSearchSkill.SKILL_VERSION == "1.0.0"

    def test_supports_multiple_instances(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        assert NativeVectorSearchSkill.SUPPORTS_MULTIPLE_INSTANCES is True

    def test_required_packages_empty(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        assert NativeVectorSearchSkill.REQUIRED_PACKAGES == []

    def test_required_env_vars_empty(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        assert NativeVectorSearchSkill.REQUIRED_ENV_VARS == []


class TestParameterSchema:
    """Verify the parameter schema returned by get_parameter_schema."""

    def test_schema_has_expected_keys(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        schema = NativeVectorSearchSkill.get_parameter_schema()

        expected_keys = [
            "index_file", "build_index", "source_dir", "remote_url",
            "index_name", "count", "similarity_threshold", "tags",
            "global_tags", "file_types", "exclude_patterns",
            "no_results_message", "response_prefix", "response_postfix",
            "max_content_length", "response_format_callback", "description",
            "hints", "nlp_backend", "query_nlp_backend", "index_nlp_backend",
            "backend", "connection_string", "collection_name", "verbose",
            "keyword_weight", "model_name", "overwrite",
            # inherited from SkillBase
            "swaig_fields", "tool_name",
        ]
        for key in expected_keys:
            assert key in schema, f"Missing schema key: {key}"

    def test_schema_count_defaults_to_five(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        schema = NativeVectorSearchSkill.get_parameter_schema()
        assert schema["count"]["default"] == 5

    def test_schema_backend_enum(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        schema = NativeVectorSearchSkill.get_parameter_schema()
        assert set(schema["backend"]["enum"]) == {"sqlite", "pgvector"}

    def test_schema_nlp_backend_enum(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        schema = NativeVectorSearchSkill.get_parameter_schema()
        assert set(schema["nlp_backend"]["enum"]) == {"basic", "spacy", "nltk"}

    def test_schema_model_name_default(self):
        from signalwire_agents.skills.native_vector_search.skill import NativeVectorSearchSkill
        schema = NativeVectorSearchSkill.get_parameter_schema()
        assert schema["model_name"]["default"] == "mini"


# ===========================================================================
# get_instance_key
# ===========================================================================

class TestGetInstanceKey:
    """Test the get_instance_key method."""

    def test_default_instance_key(self):
        skill = _make_skill()
        key = skill.get_instance_key()
        assert key == "native_vector_search_search_knowledge_default"

    def test_custom_tool_name_and_index_file(self):
        skill = _make_skill({"tool_name": "my_tool", "index_file": "/tmp/test.swsearch"})
        key = skill.get_instance_key()
        assert key == "native_vector_search_my_tool_/tmp/test.swsearch"


# ===========================================================================
# setup() -- remote mode
# ===========================================================================

class TestSetupRemoteMode:
    """Test setup() when remote_url is configured."""

    @patch("signalwire_agents.skills.native_vector_search.skill.requests", create=True)
    def test_remote_setup_success(self, mock_requests_mod):
        """Successful health check sets use_remote=True and search_available=True."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch.dict("sys.modules", {"requests": mock_requests_mod}):
            mock_requests_mod.get.return_value = mock_response

            skill = _make_skill({"remote_url": "http://localhost:8001"})
            result = skill.setup()

        assert result is True
        assert skill.use_remote is True
        assert skill.search_available is True
        assert skill.search_engine is None

    @patch("signalwire_agents.skills.native_vector_search.skill.requests", create=True)
    def test_remote_setup_auth_failure(self, mock_requests_mod):
        """401 from remote server means search_available=False."""
        mock_response = Mock()
        mock_response.status_code = 401

        with patch.dict("sys.modules", {"requests": mock_requests_mod}):
            mock_requests_mod.get.return_value = mock_response

            skill = _make_skill({"remote_url": "http://localhost:8001"})
            result = skill.setup()

        assert result is False
        assert skill.search_available is False

    @patch("signalwire_agents.skills.native_vector_search.skill.requests", create=True)
    def test_remote_setup_non_200_status(self, mock_requests_mod):
        """Non-200 and non-401 status returns False."""
        mock_response = Mock()
        mock_response.status_code = 500

        with patch.dict("sys.modules", {"requests": mock_requests_mod}):
            mock_requests_mod.get.return_value = mock_response

            skill = _make_skill({"remote_url": "http://localhost:8001"})
            result = skill.setup()

        assert result is False
        assert skill.search_available is False

    def test_remote_setup_connection_error(self):
        """Connection failure returns False."""
        mock_requests_mod = Mock()
        mock_requests_mod.get.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"requests": mock_requests_mod}):
            skill = _make_skill({"remote_url": "http://localhost:8001"})
            result = skill.setup()

        assert result is False
        assert skill.search_available is False

    def test_remote_url_auth_parsing(self):
        """Credentials embedded in URL are extracted correctly."""
        mock_requests_mod = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests_mod.get.return_value = mock_response

        with patch.dict("sys.modules", {"requests": mock_requests_mod}):
            skill = _make_skill({"remote_url": "http://user:pass@localhost:8001/api"})
            skill.setup()

        assert skill.remote_auth == ("user", "pass")
        assert "user" not in skill.remote_base_url
        assert "pass" not in skill.remote_base_url
        assert "localhost:8001" in skill.remote_base_url
        assert "/api" in skill.remote_base_url

    def test_remote_url_without_auth(self):
        """URL without credentials sets remote_auth to None."""
        mock_requests_mod = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests_mod.get.return_value = mock_response

        with patch.dict("sys.modules", {"requests": mock_requests_mod}):
            skill = _make_skill({"remote_url": "http://localhost:8001"})
            skill.setup()

        assert skill.remote_auth is None
        assert skill.remote_base_url == "http://localhost:8001"


# ===========================================================================
# setup() -- local mode (no remote_url)
# ===========================================================================

class TestSetupLocalMode:
    """Test setup() when no remote_url is set (local mode)."""

    def test_local_setup_search_import_failure(self):
        """When search dependencies are missing, setup still returns True."""
        with patch.dict("sys.modules", {
            "signalwire_agents.search": None,
        }):
            with patch(
                "signalwire_agents.skills.native_vector_search.skill.NativeVectorSearchSkill.setup"
            ) as _:
                # We need to test the actual method, so call it manually
                pass

        # Simpler approach: just call setup and mock the import inside
        skill = _make_skill()
        with patch("builtins.__import__", side_effect=_import_raiser("signalwire_agents.search")):
            result = skill.setup()

        assert result is True
        assert skill.search_available is False
        assert skill.use_remote is False

    def test_local_setup_default_params(self):
        """Default local setup populates expected attributes."""
        skill = _make_skill()
        with patch("builtins.__import__", side_effect=_import_raiser("signalwire_agents.search")):
            skill.setup()

        assert skill.tool_name == "search_knowledge"
        assert skill.backend == "sqlite"
        assert skill.count == 5
        assert skill.similarity_threshold == 0.0
        assert skill.tags == []
        assert skill.max_content_length == 32768
        assert skill.model_name == "mini"
        assert skill.use_remote is False

    def test_local_setup_custom_params(self):
        """Custom params are correctly propagated."""
        params = {
            "tool_name": "custom_search",
            "count": 10,
            "similarity_threshold": 0.5,
            "tags": ["docs", "api"],
            "no_results_message": "Nothing found for '{query}'",
            "response_prefix": "PREFIX:",
            "response_postfix": ":POSTFIX",
            "max_content_length": 16000,
            "model_name": "base",
        }
        skill = _make_skill(params)
        with patch("builtins.__import__", side_effect=_import_raiser("signalwire_agents.search")):
            skill.setup()

        assert skill.tool_name == "custom_search"
        assert skill.count == 10
        assert skill.similarity_threshold == 0.5
        assert skill.tags == ["docs", "api"]
        assert skill.response_prefix == "PREFIX:"
        assert skill.response_postfix == ":POSTFIX"
        assert skill.max_content_length == 16000
        assert skill.model_name == "base"

    def test_deprecated_nlp_backend_warning(self):
        """Using deprecated 'nlp_backend' param triggers a warning and applies to both backends."""
        skill = _make_skill({"nlp_backend": "spacy"})
        with patch("builtins.__import__", side_effect=_import_raiser("signalwire_agents.search")):
            skill.setup()

        assert skill.index_nlp_backend == "spacy"
        assert skill.query_nlp_backend == "spacy"

    def test_invalid_nlp_backend_fallback(self):
        """Invalid NLP backend names fall back to 'basic'."""
        skill = _make_skill({
            "index_nlp_backend": "invalid_backend",
            "query_nlp_backend": "another_invalid",
        })
        with patch("builtins.__import__", side_effect=_import_raiser("signalwire_agents.search")):
            skill.setup()

        assert skill.index_nlp_backend == "basic"
        assert skill.query_nlp_backend == "basic"

    def test_local_setup_sqlite_with_existing_index(self):
        """When index_file exists and search is available, SearchEngine is initialized."""
        mock_search_engine = Mock()
        mock_search_engine_cls = Mock(return_value=mock_search_engine)
        mock_search_engine.config = {"embedding_model": "test-model"}

        mock_search_mod = Mock()
        mock_search_mod.SearchEngine = mock_search_engine_cls
        mock_search_mod.IndexBuilder = Mock()

        mock_query_processor = Mock()
        mock_query_processor.preprocess_query = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            with patch("os.path.exists", return_value=True):
                skill = _make_skill({"index_file": "/tmp/test.swsearch"})
                result = skill.setup()

        assert result is True
        assert skill.search_available is True
        assert skill.search_engine is mock_search_engine

    def test_local_setup_sqlite_index_not_found(self):
        """When index_file does not exist, search_engine remains None."""
        mock_search_mod = Mock()
        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            with patch("os.path.exists", return_value=False):
                skill = _make_skill({"index_file": "/tmp/nonexistent.swsearch"})
                result = skill.setup()

        assert result is True
        assert skill.search_engine is None

    def test_local_setup_pgvector_success(self):
        """pgvector backend initialises SearchEngine with connection params."""
        mock_search_engine = Mock()
        mock_search_engine_cls = Mock(return_value=mock_search_engine)

        mock_search_mod = Mock()
        mock_search_mod.SearchEngine = mock_search_engine_cls
        mock_search_mod.IndexBuilder = Mock()

        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            skill = _make_skill({
                "backend": "pgvector",
                "connection_string": "postgresql://user:pass@localhost:5432/db",
                "collection_name": "my_collection",
            })
            result = skill.setup()

        assert result is True
        assert skill.search_available is True
        assert skill.search_engine is mock_search_engine
        mock_search_engine_cls.assert_called_once_with(
            backend="pgvector",
            connection_string="postgresql://user:pass@localhost:5432/db",
            collection_name="my_collection",
        )

    def test_local_setup_pgvector_missing_params(self):
        """pgvector backend without connection_string or collection_name sets search_available=False."""
        mock_search_mod = Mock()
        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            skill = _make_skill({"backend": "pgvector"})
            result = skill.setup()

        assert result is True
        assert skill.search_available is False

    def test_local_setup_pgvector_connection_failure(self):
        """pgvector SearchEngine init failure sets search_available=False."""
        mock_search_mod = Mock()
        mock_search_mod.SearchEngine = Mock(side_effect=Exception("Connection refused"))
        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            skill = _make_skill({
                "backend": "pgvector",
                "connection_string": "postgresql://localhost/db",
                "collection_name": "col",
            })
            result = skill.setup()

        assert result is True
        assert skill.search_available is False


# ===========================================================================
# setup() -- auto-build index
# ===========================================================================

class TestSetupAutoBuild:
    """Test setup() when build_index is True."""

    def test_auto_build_sqlite_generates_index_name(self):
        """When build_index=True and no index_file, filename is derived from source_dir."""
        mock_builder = Mock()
        mock_builder_cls = Mock(return_value=mock_builder)
        mock_resolve = Mock(return_value="sentence-transformers/all-MiniLM-L6-v2")

        mock_search_mod = Mock()
        mock_search_mod.IndexBuilder = mock_builder_cls
        mock_search_mod.SearchEngine = Mock()

        mock_models = Mock()
        mock_models.resolve_model_alias = mock_resolve

        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.models": mock_models,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            with patch("os.path.exists", return_value=False):
                skill = _make_skill({
                    "build_index": True,
                    "source_dir": "/data/my_docs",
                })
                skill.setup()

        # index_file should be derived from source_dir name
        assert skill.index_file == "my_docs.swsearch"
        mock_builder.build_index.assert_called_once()

    def test_auto_build_sqlite_skips_existing_index(self):
        """When the index file already exists, build is skipped."""
        mock_search_mod = Mock()
        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            with patch("os.path.exists", return_value=True):
                skill = _make_skill({
                    "build_index": True,
                    "source_dir": "/data/docs",
                    "index_file": "/tmp/existing.swsearch",
                })
                skill.setup()

        # IndexBuilder should NOT have been called since index already exists
        # (the import of IndexBuilder for building only happens when file does not exist)
        assert skill.search_available is True

    def test_auto_build_sqlite_failure(self):
        """Build failure sets search_available=False but setup still returns True."""
        mock_builder = Mock()
        mock_builder.build_index.side_effect = Exception("Build failed")
        mock_builder_cls = Mock(return_value=mock_builder)
        mock_resolve = Mock(return_value="mini")

        mock_search_mod = Mock()
        mock_search_mod.IndexBuilder = mock_builder_cls
        mock_search_mod.SearchEngine = Mock()

        mock_models = Mock()
        mock_models.resolve_model_alias = mock_resolve

        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.models": mock_models,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            with patch("os.path.exists", return_value=False):
                skill = _make_skill({
                    "build_index": True,
                    "source_dir": "/data/docs",
                    "index_file": "/tmp/test.swsearch",
                })
                result = skill.setup()

        assert result is True
        assert skill.search_available is False

    def test_auto_build_pgvector(self):
        """pgvector auto-build calls IndexBuilder with correct backend params."""
        mock_builder = Mock()
        mock_builder_cls = Mock(return_value=mock_builder)
        mock_resolve = Mock(return_value="base-model")

        mock_search_mod = Mock()
        mock_search_mod.IndexBuilder = mock_builder_cls
        mock_search_mod.SearchEngine = Mock()

        mock_models = Mock()
        mock_models.resolve_model_alias = mock_resolve

        mock_query_processor = Mock()

        with patch.dict("sys.modules", {
            "signalwire_agents.search": mock_search_mod,
            "signalwire_agents.search.models": mock_models,
            "signalwire_agents.search.query_processor": mock_query_processor,
        }):
            skill = _make_skill({
                "build_index": True,
                "source_dir": "/data/docs",
                "backend": "pgvector",
                "connection_string": "postgresql://localhost/db",
                "collection_name": "my_col",
            })
            skill.setup()

        mock_builder_cls.assert_called_once_with(
            backend="pgvector",
            connection_string="postgresql://localhost/db",
            model_name="base-model",
            verbose=False,
            index_nlp_backend="nltk",
        )
        mock_builder.build_index.assert_called_once()


# ===========================================================================
# register_tools()
# ===========================================================================

class TestRegisterTools:
    """Test register_tools method."""

    def _setup_skill_for_register(self, params=None):
        """Helper to create a skill ready for register_tools."""
        skill = _make_skill(params or {})
        # Manually set attributes that setup() would set
        skill.tool_name = params.get("tool_name", "search_knowledge") if params else "search_knowledge"
        skill.count = params.get("count", 5) if params else 5
        skill.use_remote = False
        return skill

    def test_register_tools_defines_tool(self):
        """register_tools calls define_tool with expected parameters."""
        skill = self._setup_skill_for_register()
        skill.register_tools()

        skill.agent.define_tool.assert_called_once()
        call_kwargs = skill.agent.define_tool.call_args
        # define_tool is called via self.define_tool which delegates to agent.define_tool
        # Check that the tool name and handler are present
        assert call_kwargs.kwargs.get("name") or call_kwargs[1].get("name") == "search_knowledge"

    def test_register_tools_creates_knowledge_search_section(self):
        """When section does not exist, a new one is created."""
        skill = self._setup_skill_for_register()
        skill.agent.prompt_has_section.return_value = False

        skill.register_tools()

        skill.agent.prompt_add_section.assert_called_once()
        call_kwargs = skill.agent.prompt_add_section.call_args[1]
        assert call_kwargs["title"] == "Knowledge Search"

    def test_register_tools_adds_to_existing_section(self):
        """When section already exists, a bullet is added instead."""
        skill = self._setup_skill_for_register()
        skill.agent.prompt_has_section.return_value = True

        skill.register_tools()

        skill.agent.prompt_add_to_section.assert_called_once()
        call_kwargs = skill.agent.prompt_add_to_section.call_args[1]
        assert call_kwargs["title"] == "Knowledge Search"

    def test_register_tools_custom_description(self):
        """Custom description is passed to define_tool."""
        skill = self._setup_skill_for_register({"description": "Search my docs"})
        skill.register_tools()

        call_kwargs = skill.agent.define_tool.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert kwargs.get("description") == "Search my docs"


# ===========================================================================
# _search_handler()
# ===========================================================================

class TestSearchHandler:
    """Test the _search_handler method."""

    def _setup_skill_for_search(self, **overrides):
        """Create a skill with search attributes pre-configured."""
        skill = _make_skill()
        skill.search_available = True
        skill.use_remote = False
        skill.search_engine = Mock()
        skill.tool_name = "search_knowledge"
        skill.index_file = None
        skill.count = 5
        skill.similarity_threshold = 0.0
        skill.tags = []
        skill.keyword_weight = None
        skill.no_results_message = "No information found for '{query}'"
        skill.response_prefix = ""
        skill.response_postfix = ""
        skill.max_content_length = 32768
        skill.response_format_callback = None
        skill.remote_url = None
        skill.remote_base_url = None
        skill.query_nlp_backend = "basic"

        for k, v in overrides.items():
            setattr(skill, k, v)
        return skill

    def test_search_unavailable(self):
        """When search is not available, return an error message."""
        skill = self._setup_skill_for_search(search_available=False, import_error="missing dep")
        result = skill._search_handler({"query": "test"}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "not available" in result.response.lower()

    def test_search_engine_missing(self):
        """When search_engine is None in local mode, return an error."""
        skill = self._setup_skill_for_search(search_engine=None)
        result = skill._search_handler({"query": "test"}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "not available" in result.response.lower()

    def test_empty_query(self):
        """Empty query string returns a prompt to provide a query."""
        skill = self._setup_skill_for_search()
        result = skill._search_handler({"query": ""}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_whitespace_only_query(self):
        """Whitespace-only query is treated as empty."""
        skill = self._setup_skill_for_search()
        result = skill._search_handler({"query": "   "}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_missing_query_key(self):
        """Missing 'query' key is treated as empty."""
        skill = self._setup_skill_for_search()
        result = skill._search_handler({}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_local_search_no_results(self):
        """Local search returning no results uses no_results_message."""
        mock_preprocess = Mock(return_value={"enhanced_text": "test", "vector": [0.1, 0.2]})
        skill = self._setup_skill_for_search()
        skill.search_engine.search.return_value = []

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test"}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "No information found for 'test'" in result.response

    def test_local_search_no_results_with_prefix_postfix(self):
        """Prefix and postfix are added to the no-results message."""
        mock_preprocess = Mock(return_value={"enhanced_text": "test", "vector": []})
        skill = self._setup_skill_for_search(
            response_prefix="[START]",
            response_postfix="[END]",
        )
        skill.search_engine.search.return_value = []

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test"}, {})

        assert result.response.startswith("[START]")
        assert result.response.endswith("[END]")

    def test_local_search_with_results(self):
        """Successful local search formats results correctly."""
        mock_preprocess = Mock(return_value={"enhanced_text": "test query", "vector": [0.1]})
        skill = self._setup_skill_for_search()
        skill.search_engine.search.return_value = [
            {
                "content": "This is the answer.",
                "score": 0.95,
                "metadata": {"filename": "doc.md", "section": "Overview"},
                "tags": ["docs"],
            },
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test query"}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "Found 1 relevant results" in result.response
        assert "doc.md" in result.response
        assert "Overview" in result.response
        assert "This is the answer." in result.response
        assert "0.95" in result.response

    def test_local_search_with_multiple_results(self):
        """Multiple results are all formatted and included."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})
        skill = self._setup_skill_for_search()
        skill.search_engine.search.return_value = [
            {"content": "Answer 1", "score": 0.9, "metadata": {"filename": "a.md"}, "tags": []},
            {"content": "Answer 2", "score": 0.8, "metadata": {"filename": "b.md"}, "tags": []},
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "q"}, {})

        assert "Found 2 relevant results" in result.response
        assert "Result 1" in result.response
        assert "Result 2" in result.response

    def test_local_search_content_truncation(self):
        """Long content is truncated to per-result limit."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})
        skill = self._setup_skill_for_search(max_content_length=1500)
        long_content = "x" * 5000
        skill.search_engine.search.return_value = [
            {"content": long_content, "score": 0.9, "metadata": {"filename": "a.md"}, "tags": []},
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "q"}, {})

        # The content should be truncated and end with "..."
        assert "..." in result.response
        assert len(result.response) < 5000

    def test_local_search_with_prefix_postfix(self):
        """Prefix and postfix are included in the response."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})
        skill = self._setup_skill_for_search(
            response_prefix="<<PREFIX>>",
            response_postfix="<<POSTFIX>>",
        )
        skill.search_engine.search.return_value = [
            {"content": "Answer", "score": 0.9, "metadata": {"filename": "a.md"}, "tags": []},
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "q"}, {})

        assert "<<PREFIX>>" in result.response
        assert "<<POSTFIX>>" in result.response

    def test_local_search_count_override(self):
        """The count argument from args overrides default count."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})
        skill = self._setup_skill_for_search()
        skill.search_engine.search.return_value = []
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            skill._search_handler({"query": "q", "count": 3}, {})

        # Check that search was called with count=3
        call_kwargs = skill.search_engine.search.call_args[1]
        assert call_kwargs["count"] == 3

    def test_search_exception_handling_generic(self):
        """Generic exceptions return a user-friendly message."""
        mock_preprocess = Mock(side_effect=RuntimeError("unexpected"))
        skill = self._setup_skill_for_search()
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test"}, {})

        assert isinstance(result, SwaigFunctionResult)
        assert "sorry" in result.response.lower()
        assert "rephrasing" in result.response.lower()

    def test_search_exception_handling_nltk(self):
        """NLTK-related exceptions provide specific guidance."""
        mock_preprocess = Mock(side_effect=RuntimeError("punkt resource missing"))
        skill = self._setup_skill_for_search()
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test"}, {})

        assert "language processing" in result.response.lower()

    def test_search_exception_handling_vector(self):
        """Vector/embedding exceptions provide specific guidance."""
        mock_preprocess = Mock(side_effect=RuntimeError("vector dimension mismatch"))
        skill = self._setup_skill_for_search()
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test"}, {})

        assert "indexing" in result.response.lower()

    def test_search_exception_handling_timeout(self):
        """Timeout/connection exceptions provide specific guidance."""
        mock_preprocess = Mock(side_effect=RuntimeError("connection timeout"))
        skill = self._setup_skill_for_search()
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "test"}, {})

        assert "temporarily unavailable" in result.response.lower()

    def test_response_format_callback_with_results(self):
        """Custom response_format_callback transforms the response."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})

        def my_callback(**kwargs):
            return f"CUSTOM: {kwargs['query']}"

        skill = self._setup_skill_for_search(response_format_callback=my_callback)
        skill.search_engine.search.return_value = [
            {"content": "Answer", "score": 0.9, "metadata": {"filename": "a.md"}, "tags": []},
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "hello"}, {})

        assert result.response == "CUSTOM: hello"

    def test_response_format_callback_with_no_results(self):
        """Custom callback is also called when there are no results."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})

        def my_callback(**kwargs):
            return "CUSTOM NO RESULTS"

        skill = self._setup_skill_for_search(response_format_callback=my_callback)
        skill.search_engine.search.return_value = []
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "hello"}, {})

        assert result.response == "CUSTOM NO RESULTS"

    def test_response_format_callback_non_string_ignored(self):
        """Callback returning non-string is ignored."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})

        def bad_callback(**kwargs):
            return 12345  # not a string

        skill = self._setup_skill_for_search(response_format_callback=bad_callback)
        skill.search_engine.search.return_value = [
            {"content": "Answer", "score": 0.9, "metadata": {"filename": "a.md"}, "tags": []},
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "hello"}, {})

        # Should fall back to the original formatted response
        assert "Found 1 relevant results" in result.response

    def test_response_format_callback_exception(self):
        """Exception in callback is caught and original response is used."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})

        def exploding_callback(**kwargs):
            raise ValueError("boom")

        skill = self._setup_skill_for_search(response_format_callback=exploding_callback)
        skill.search_engine.search.return_value = [
            {"content": "Answer", "score": 0.9, "metadata": {"filename": "a.md"}, "tags": []},
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "hello"}, {})

        # Should still have a valid response
        assert "Found 1 relevant results" in result.response

    def test_tags_from_metadata_nested(self):
        """Tags are extracted from nested metadata structure."""
        mock_preprocess = Mock(return_value={"enhanced_text": "q", "vector": [0.1]})
        skill = self._setup_skill_for_search()
        skill.search_engine.search.return_value = [
            {
                "content": "Answer",
                "score": 0.9,
                "metadata": {
                    "filename": "a.md",
                    "tags": ["tag1", "tag2"],
                },
                "tags": [],
            },
        ]
        skill.search_engine.config = {}

        with patch.dict("sys.modules", {
            "signalwire_agents.search.query_processor": Mock(preprocess_query=mock_preprocess),
        }):
            result = skill._search_handler({"query": "q"}, {})

        assert "tag1" in result.response
        assert "tag2" in result.response


# ===========================================================================
# _search_remote()
# ===========================================================================

class TestSearchRemote:
    """Test the _search_remote method."""

    def _setup_remote_skill(self):
        """Create a skill configured for remote mode."""
        skill = _make_skill()
        skill.remote_base_url = "http://localhost:8001"
        skill.remote_auth = None
        skill.index_name = "default"
        skill.similarity_threshold = 0.0
        skill.tags = []
        return skill

    def test_remote_search_success(self):
        """Successful remote search returns formatted results."""
        mock_requests = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "content": "Remote result",
                    "score": 0.85,
                    "metadata": {"filename": "remote.md"},
                }
            ]
        }
        mock_requests.post.return_value = mock_response

        skill = self._setup_remote_skill()

        with patch.dict("sys.modules", {"requests": mock_requests}):
            results = skill._search_remote("test query", None, 5)

        assert len(results) == 1
        assert results[0]["content"] == "Remote result"
        assert results[0]["score"] == 0.85

    def test_remote_search_failure_status(self):
        """Non-200 status from remote returns empty list."""
        mock_requests = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests.post.return_value = mock_response

        skill = self._setup_remote_skill()

        with patch.dict("sys.modules", {"requests": mock_requests}):
            results = skill._search_remote("test", None, 5)

        assert results == []

    def test_remote_search_exception(self):
        """Exception during remote search returns empty list."""
        mock_requests = Mock()
        mock_requests.post.side_effect = ConnectionError("refused")

        skill = self._setup_remote_skill()

        with patch.dict("sys.modules", {"requests": mock_requests}):
            results = skill._search_remote("test", None, 5)

        assert results == []

    def test_remote_search_with_auth(self):
        """Auth credentials are passed to the remote request."""
        mock_requests = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_requests.post.return_value = mock_response

        skill = self._setup_remote_skill()
        skill.remote_auth = ("user", "pass")

        with patch.dict("sys.modules", {"requests": mock_requests}):
            skill._search_remote("test", None, 5)

        call_kwargs = mock_requests.post.call_args[1]
        assert call_kwargs["auth"] == ("user", "pass")

    def test_remote_search_sends_correct_payload(self):
        """Remote search sends the expected JSON payload."""
        mock_requests = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_requests.post.return_value = mock_response

        skill = self._setup_remote_skill()
        skill.index_name = "my_index"
        skill.similarity_threshold = 0.3
        skill.tags = ["api"]

        with patch.dict("sys.modules", {"requests": mock_requests}):
            skill._search_remote("my query", None, 10)

        call_kwargs = mock_requests.post.call_args
        json_body = call_kwargs[1]["json"]
        assert json_body["query"] == "my query"
        assert json_body["index_name"] == "my_index"
        assert json_body["count"] == 10
        assert json_body["similarity_threshold"] == 0.3
        assert json_body["tags"] == ["api"]

    def test_remote_search_empty_results_array(self):
        """Remote search with empty results array returns empty list."""
        mock_requests = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_requests.post.return_value = mock_response

        skill = self._setup_remote_skill()

        with patch.dict("sys.modules", {"requests": mock_requests}):
            results = skill._search_remote("test", None, 5)

        assert results == []


# ===========================================================================
# _search_handler() with remote mode
# ===========================================================================

class TestSearchHandlerRemoteMode:
    """Test _search_handler when use_remote=True."""

    def _setup_remote_skill_for_search(self):
        """Create a skill configured for remote search handling."""
        skill = _make_skill()
        skill.search_available = True
        skill.use_remote = True
        skill.search_engine = None
        skill.tool_name = "search_knowledge"
        skill.count = 5
        skill.similarity_threshold = 0.0
        skill.tags = []
        skill.keyword_weight = None
        skill.no_results_message = "No information found for '{query}'"
        skill.response_prefix = ""
        skill.response_postfix = ""
        skill.max_content_length = 32768
        skill.response_format_callback = None
        skill.remote_url = "http://localhost:8001"
        skill.remote_base_url = "http://localhost:8001"
        skill.remote_auth = None
        skill.index_name = "default"
        return skill

    def test_remote_handler_calls_search_remote(self):
        """Handler in remote mode calls _search_remote."""
        skill = self._setup_remote_skill_for_search()
        skill._search_remote = Mock(return_value=[
            {"content": "Result", "score": 0.9, "metadata": {"filename": "r.md"}, "tags": []},
        ])

        result = skill._search_handler({"query": "test"}, {})

        skill._search_remote.assert_called_once_with("test", None, 5)
        assert "Found 1 relevant results" in result.response

    def test_remote_handler_no_results(self):
        """Remote handler with no results returns no_results_message."""
        skill = self._setup_remote_skill_for_search()
        skill._search_remote = Mock(return_value=[])

        result = skill._search_handler({"query": "test"}, {})

        assert "No information found for 'test'" in result.response


# ===========================================================================
# get_hints(), get_global_data(), get_prompt_sections(), cleanup()
# ===========================================================================

class TestMiscMethods:
    """Test auxiliary methods on the skill."""

    def test_get_hints_defaults(self):
        skill = _make_skill()
        hints = skill.get_hints()
        assert "search" in hints
        assert "find" in hints
        assert "look up" in hints
        assert "documentation" in hints
        assert "knowledge base" in hints

    def test_get_hints_custom(self):
        skill = _make_skill({"hints": ["custom1", "custom2"]})
        hints = skill.get_hints()
        assert "custom1" in hints
        assert "custom2" in hints
        # defaults are still present
        assert "search" in hints

    def test_get_global_data_no_engine(self):
        skill = _make_skill()
        skill.search_engine = None
        data = skill.get_global_data()
        assert data == {}

    def test_get_global_data_with_engine(self):
        skill = _make_skill()
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {"documents": 100}
        skill.search_engine = mock_engine

        data = skill.get_global_data()
        assert data["search_stats"]["documents"] == 100

    def test_get_global_data_engine_error(self):
        skill = _make_skill()
        mock_engine = Mock()
        mock_engine.get_stats.side_effect = Exception("stats error")
        skill.search_engine = mock_engine

        data = skill.get_global_data()
        assert data == {}

    def test_get_prompt_sections_returns_empty(self):
        skill = _make_skill()
        assert skill.get_prompt_sections() == []

    def test_cleanup_no_temp_dirs(self):
        """cleanup with no _temp_dirs attribute does nothing."""
        skill = _make_skill()
        # Should not raise
        skill.cleanup()

    def test_cleanup_with_temp_dirs(self):
        """cleanup removes temp directories."""
        skill = _make_skill()
        skill._temp_dirs = ["/tmp/fake_dir1", "/tmp/fake_dir2"]

        with patch("shutil.rmtree") as mock_rmtree:
            skill.cleanup()

        assert mock_rmtree.call_count == 2

    def test_cleanup_rmtree_error_ignored(self):
        """cleanup ignores errors from rmtree."""
        skill = _make_skill()
        skill._temp_dirs = ["/tmp/fake_dir"]

        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            # Should not raise
            skill.cleanup()


# ===========================================================================
# _add_prompt_section()
# ===========================================================================

class TestAddPromptSection:
    """Test _add_prompt_section method."""

    def test_add_prompt_section_success(self):
        skill = _make_skill()
        skill.tool_name = "my_search"
        mock_agent = Mock()

        skill._add_prompt_section(mock_agent)

        mock_agent.prompt_add_section.assert_called_once()
        call_kwargs = mock_agent.prompt_add_section.call_args[1]
        assert call_kwargs["title"] == "Local Document Search"
        assert "my_search" in call_kwargs["body"]

    def test_add_prompt_section_error_handled(self):
        skill = _make_skill()
        skill.tool_name = "search"
        mock_agent = Mock()
        mock_agent.prompt_add_section.side_effect = Exception("prompt error")

        # Should not raise
        skill._add_prompt_section(mock_agent)


# ===========================================================================
# Helpers
# ===========================================================================

def _import_raiser(blocked_module):
    """
    Return a side_effect function for patching builtins.__import__ that raises
    ImportError for a specific module while allowing everything else.
    """
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _custom_import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"Mocked import error for {name}")
        return real_import(name, *args, **kwargs)

    return _custom_import
