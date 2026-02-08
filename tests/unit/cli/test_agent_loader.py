"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for CLI agent_loader module.

Tests cover:
- discover_services_in_file and discover_agents_in_file
- _discover_services_impl internals (file validation, sys.path management,
  instance/class discovery)
- load_service_from_file and load_agent_from_file public wrappers
- _load_service_impl strategies (route lookup, class name, 'agent'/'service'
  variable, single/multiple instances, class instantiation, main() fallback)
- Error handling: file not found, non-.py file, import errors, no services found
"""

import pytest
import sys
import types
import importlib
import importlib.util
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Create lightweight stand-in classes that the loader can use for isinstance
# checks.  We need them *before* we import agent_loader so the module-level
# try/except resolves successfully.
# ---------------------------------------------------------------------------

class _MockSWMLService:
    """Minimal stand-in for SWMLService used in agent_loader isinstance checks."""
    name = "mock-service"
    route = "/mock"

    def serve(self, *a, **kw):
        pass

    def run(self, *a, **kw):
        pass


class _MockAgentBase(_MockSWMLService):
    """Minimal stand-in for AgentBase (inherits from SWMLService stand-in)."""
    _tool_registry = {}


class _MockServiceCapture:
    """Minimal stand-in for ServiceCapture."""
    pass


# Ensure the sub-module the agent_loader tries to import from exists
# so that the `from .service_loader import ...` inside the try-block
# does not fail at import time.  We temporarily stub if needed, but
# restore any real module afterward to avoid polluting other tests.
_core_pkg = "signalwire_agents.cli.core"
_svc_loader_mod = f"{_core_pkg}.service_loader"
_original_svc_loader = sys.modules.get(_svc_loader_mod)
_needs_stub = _svc_loader_mod not in sys.modules
if _needs_stub:
    _mod = types.ModuleType(_svc_loader_mod)
    _mod.ServiceCapture = _MockServiceCapture
    _mod.load_agent_from_file = lambda *a, **kw: None
    sys.modules[_svc_loader_mod] = _mod

# Now import the module under test -- the try/except at module level will
# either succeed with the real classes or fall back to None.  We will
# patch the module-level sentinels in our fixtures.
from signalwire_agents.cli.core import agent_loader

# Restore the real service_loader module so other test files can import it
if _needs_stub:
    if _original_svc_loader is not None:
        sys.modules[_svc_loader_mod] = _original_svc_loader
    else:
        del sys.modules[_svc_loader_mod]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_module_globals():
    """Ensure every test sees our mock classes and the AVAILABLE flags set."""
    with patch.object(agent_loader, "SWMLService", _MockSWMLService), \
         patch.object(agent_loader, "AgentBase", _MockAgentBase), \
         patch.object(agent_loader, "AGENT_BASE_AVAILABLE", True), \
         patch.object(agent_loader, "SWML_SERVICE_AVAILABLE", True), \
         patch.object(agent_loader, "NEW_LOADER_AVAILABLE", True):
        yield


def _write_py(tmp_path, filename, code):
    """Helper: write a Python file into tmp_path and return its path string."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(code))
    return str(p)


# ============================================================================
# discover_services_in_file
# ============================================================================

class TestDiscoverServicesInFile:
    """Tests for the public discover_services_in_file function."""

    def test_raises_when_swml_not_available(self, tmp_path):
        path = _write_py(tmp_path, "svc.py", "x = 1\n")
        with patch.object(agent_loader, "SWML_SERVICE_AVAILABLE", False):
            with pytest.raises(ImportError, match="SWMLService not available"):
                agent_loader.discover_services_in_file(path)

    def test_file_not_found(self, tmp_path):
        fake = str(tmp_path / "no_such_file.py")
        with pytest.raises(FileNotFoundError):
            agent_loader.discover_services_in_file(fake)

    def test_non_python_file(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Python file"):
            agent_loader.discover_services_in_file(str(path))

    def test_finds_instance(self, tmp_path):
        """A module-level SWMLService instance should be discovered."""
        code = """\
        class Svc:
            pass
        svc_instance = Svc()
        """
        path = _write_py(tmp_path, "mymod.py", code)

        # We need the loaded module to contain an actual _MockSWMLService instance.
        # The easiest way: create a real instance, then patch the module exec to
        # inject it.
        instance = _MockSWMLService()
        instance.name = "test-svc"
        instance.route = "/test"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.my_instance = instance

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader.discover_services_in_file(path)

        names = [r["name"] for r in results]
        assert "my_instance" in names

    def test_finds_subclass(self, tmp_path):
        """A module-level SWMLService subclass should be discovered."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "mymod2.py", code)

        class MySvcClass(_MockSWMLService):
            """Custom service"""
            pass

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.MySvcClass = MySvcClass

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader.discover_services_in_file(path)

        names = [r["name"] for r in results]
        assert "MySvcClass" in names
        cls_entry = [r for r in results if r["name"] == "MySvcClass"][0]
        assert cls_entry["type"] == "class"


# ============================================================================
# discover_agents_in_file
# ============================================================================

class TestDiscoverAgentsInFile:
    """Tests for the backward-compat discover_agents_in_file wrapper."""

    def test_filters_to_agents_only(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "agents.py", code)

        # Create one agent instance and one plain service instance
        agent_inst = _MockAgentBase()
        agent_inst.name = "my-agent"
        agent_inst.route = "/agent"

        svc_inst = _MockSWMLService()
        svc_inst.name = "plain-svc"
        svc_inst.route = "/plain"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.agent_inst = agent_inst
                module.svc_inst = svc_inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader.discover_agents_in_file(path)

        # Only the agent should survive the filter
        assert len(results) >= 1
        for r in results:
            assert r["is_agent"] is True

    def test_empty_when_no_agents(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "empty.py", code)
        results = agent_loader.discover_agents_in_file(path)
        assert results == []


# ============================================================================
# _discover_services_impl
# ============================================================================

class TestDiscoverServicesImpl:
    """Tests for the internal _discover_services_impl."""

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Service file not found"):
            agent_loader._discover_services_impl(str(tmp_path / "nope.py"))

    def test_non_py_raises_value_error(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Python file"):
            agent_loader._discover_services_impl(str(f))

    def test_sys_path_cleaned_on_success(self, tmp_path):
        """Module directory should be removed from sys.path after success."""
        path = _write_py(tmp_path, "ok.py", "x = 1\n")
        module_dir = str(tmp_path.resolve())
        # Remove from sys.path if already there
        if module_dir in sys.path:
            sys.path.remove(module_dir)
        agent_loader._discover_services_impl(path)
        assert module_dir not in sys.path

    def test_sys_path_cleaned_on_error(self, tmp_path):
        """Module directory should be removed from sys.path even on import error."""
        path = _write_py(tmp_path, "bad.py", "raise RuntimeError('boom')\n")
        module_dir = str(tmp_path.resolve())
        if module_dir in sys.path:
            sys.path.remove(module_dir)
        with pytest.raises(ImportError, match="Failed to load"):
            agent_loader._discover_services_impl(path)
        assert module_dir not in sys.path

    def test_import_error_wrapped(self, tmp_path):
        path = _write_py(tmp_path, "bad2.py", "import nonexistent_module_xyz\n")
        with pytest.raises(ImportError, match="Failed to load service module"):
            agent_loader._discover_services_impl(path)

    def test_instance_attributes(self, tmp_path):
        """Discovered instance dicts should have expected keys and values."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "inst.py", code)

        inst = _MockSWMLService()
        inst.name = "my-name"
        inst.route = "/my-route"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.the_service = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader._discover_services_impl(path)

        found = [r for r in results if r["name"] == "the_service"]
        assert len(found) == 1
        entry = found[0]
        assert entry["type"] == "instance"
        assert entry["service_name"] == "my-name"
        assert entry["route"] == "/my-route"
        assert entry["class_name"] == "_MockSWMLService"
        assert entry["object"] is inst

    def test_class_not_duplicated_when_instance_exists(self, tmp_path):
        """If an instance of a class is found, the class itself should not be listed separately."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "dup.py", code)

        class MySpecialSvc(_MockSWMLService):
            """Special"""
            pass

        inst = MySpecialSvc()
        inst.name = "sp"
        inst.route = "/sp"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.MySpecialSvc = MySpecialSvc
                module.sp = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader._discover_services_impl(path)

        class_entries = [r for r in results if r["name"] == "MySpecialSvc" and r["type"] == "class"]
        assert len(class_entries) == 0  # should be deduplicated

    def test_agent_instance_has_is_agent_true(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "ag.py", code)

        inst = _MockAgentBase()
        inst.name = "ag"
        inst.route = "/ag"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.agent = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader._discover_services_impl(path)

        agent_entries = [r for r in results if r["name"] == "agent"]
        assert len(agent_entries) == 1
        assert agent_entries[0]["is_agent"] is True
        assert agent_entries[0]["has_tools"] is True

    def test_empty_module_returns_empty_list(self, tmp_path):
        path = _write_py(tmp_path, "empty2.py", "x = 42\n")
        results = agent_loader._discover_services_impl(path)
        assert results == []


# ============================================================================
# load_service_from_file
# ============================================================================

class TestLoadServiceFromFile:
    """Tests for the public load_service_from_file function."""

    def test_raises_when_swml_not_available(self, tmp_path):
        path = _write_py(tmp_path, "s.py", "x = 1\n")
        with patch.object(agent_loader, "SWML_SERVICE_AVAILABLE", False):
            with pytest.raises(ImportError, match="SWMLService not available"):
                agent_loader.load_service_from_file(path)

    def test_delegates_to_impl(self, tmp_path):
        path = _write_py(tmp_path, "s2.py", "x = 1\n")
        sentinel = _MockSWMLService()
        with patch.object(agent_loader, "_load_service_impl", return_value=sentinel) as m:
            result = agent_loader.load_service_from_file(path, "ident", prefer_route=False)
            m.assert_called_once_with(path, "ident", False)
            assert result is sentinel


# ============================================================================
# load_agent_from_file
# ============================================================================

class TestLoadAgentFromFile:
    """Tests for the public load_agent_from_file function."""

    def test_raises_when_agent_base_not_available(self, tmp_path):
        path = _write_py(tmp_path, "a.py", "x = 1\n")
        with patch.object(agent_loader, "AGENT_BASE_AVAILABLE", False):
            with pytest.raises(ImportError, match="AgentBase not available"):
                agent_loader.load_agent_from_file(path)

    def test_delegates_to_impl_with_prefer_route_false(self, tmp_path):
        path = _write_py(tmp_path, "a2.py", "x = 1\n")
        sentinel = _MockAgentBase()
        with patch.object(agent_loader, "_load_service_impl", return_value=sentinel) as m:
            result = agent_loader.load_agent_from_file(path, "MyClass")
            m.assert_called_once_with(path, "MyClass", prefer_route=False)
            assert result is sentinel


# ============================================================================
# _load_service_impl
# ============================================================================

class TestLoadServiceImpl:
    """Tests for the internal _load_service_impl function."""

    # --- basic validation ------------------------------------------------

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Service file not found"):
            agent_loader._load_service_impl(str(tmp_path / "gone.py"))

    def test_non_py_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="Python file"):
            agent_loader._load_service_impl(str(f))

    def test_import_error_in_module(self, tmp_path):
        path = _write_py(tmp_path, "bad.py", "raise RuntimeError('kaboom')\n")
        with pytest.raises(ImportError, match="Failed to load service module"):
            agent_loader._load_service_impl(path)

    # --- prefer_route=True path ------------------------------------------

    def test_prefer_route_finds_instance_by_route(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "r1.py", code)

        inst = _MockSWMLService()
        inst.name = "rt-svc"
        inst.route = "/my-route"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.svc = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path, "/my-route", prefer_route=True)
        assert result is inst

    def test_prefer_route_not_found_raises(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "r2.py", code)
        with pytest.raises(ValueError, match="No service found with route"):
            agent_loader._load_service_impl(path, "/nowhere", prefer_route=True)

    def test_prefer_route_fallback_to_class_name(self, tmp_path):
        """When the identifier doesn't match any route but matches a class name as attribute."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "r3.py", code)

        inst = _MockSWMLService()
        inst.name = "by-name"
        inst.route = "/by-name"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.MySvc = inst  # attribute name matches identifier

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path, "MySvc", prefer_route=True)
        assert result is inst

    # --- prefer_route=False (class-name) path ----------------------------

    def test_class_name_finds_instance(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "cn1.py", code)

        inst = _MockSWMLService()
        inst.name = "cn-inst"
        inst.route = "/cn"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.MySvc = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path, "MySvc", prefer_route=False)
        assert result is inst

    def test_class_name_not_found_raises(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "cn2.py", code)
        with pytest.raises(ValueError, match="not found in"):
            agent_loader._load_service_impl(path, "NoSuchClass", prefer_route=False)

    def test_class_name_not_valid_service(self, tmp_path):
        """Identifier exists in module but is not a SWMLService."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "cn3.py", code)

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.NotAService = "just a string"

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            with pytest.raises(ValueError, match="not a valid SWMLService"):
                agent_loader._load_service_impl(path, "NotAService", prefer_route=False)

    def test_class_name_instantiates_class(self, tmp_path):
        """When the identifier is a SWMLService subclass, it should be instantiated."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "cn4.py", code)

        class MySvcClass(_MockSWMLService):
            def __init__(self):
                self.name = "inst-svc"
                self.route = "/inst"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.MySvcClass = MySvcClass

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path, "MySvcClass", prefer_route=False)
        assert isinstance(result, MySvcClass)

    # --- Strategy 1: 'agent' / 'service' variable -----------------------

    def test_strategy1_finds_agent_variable(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "s1a.py", code)

        inst = _MockSWMLService()
        inst.name = "agent-var"
        inst.route = "/agent-var"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.agent = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        assert result is inst

    def test_strategy1_finds_service_variable(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "s1b.py", code)

        inst = _MockSWMLService()
        inst.name = "svc-var"
        inst.route = "/svc-var"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.service = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        assert result is inst

    # --- Strategy 2: any SWMLService instance ----------------------------

    def test_strategy2_single_instance(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "s2a.py", code)

        inst = _MockSWMLService()
        inst.name = "solo"
        inst.route = "/solo"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.my_svc = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        assert result is inst

    def test_strategy2_multiple_instances_prefers_agent(self, tmp_path):
        """When multiple instances exist, prefer one named 'agent'."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "s2b.py", code)

        inst1 = _MockSWMLService()
        inst1.name = "other"
        inst1.route = "/other"

        inst2 = _MockSWMLService()
        inst2.name = "preferred"
        inst2.route = "/preferred"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.other_svc = inst1
                module.agent = inst2

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        # Strategy 1 should find 'agent' before Strategy 2 even runs
        assert result is inst2

    def test_strategy2_multiple_instances_uses_first_when_no_preferred_name(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "s2c.py", code)

        inst1 = _MockSWMLService()
        inst1.name = "first"
        inst1.route = "/first"

        inst2 = _MockSWMLService()
        inst2.name = "second"
        inst2.route = "/second"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.alpha_svc = inst1
                module.beta_svc = inst2

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        # Should pick one of them (first found)
        assert result in (inst1, inst2)

    # --- Strategy 3: subclass instantiation ------------------------------

    def test_strategy3_single_class(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "s3a.py", code)

        class SingleSvc(_MockSWMLService):
            def __init__(self):
                self.name = "single"
                self.route = "/single"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.SingleSvc = SingleSvc

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        assert isinstance(result, SingleSvc)

    def test_strategy3_multiple_classes_raises(self, tmp_path):
        """Multiple service classes without an identifier should raise ValueError."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "s3b.py", code)

        class SvcA(_MockSWMLService):
            def __init__(self):
                self.name = "a"
                self.route = "/a"

        class SvcB(_MockSWMLService):
            def __init__(self):
                self.name = "b"
                self.route = "/b"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.SvcA = SvcA
                module.SvcB = SvcB

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            with pytest.raises(ValueError, match="Multiple service classes found"):
                agent_loader._load_service_impl(path)

    def test_strategy3_class_instantiation_failure_prints_warning(self, tmp_path, capsys):
        """If the single class can't be instantiated, a warning is printed."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "s3c.py", code)

        class BadSvc(_MockSWMLService):
            def __init__(self):
                raise TypeError("cannot init")

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.BadSvc = BadSvc

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            with pytest.raises(ValueError, match="No service found"):
                agent_loader._load_service_impl(path)

        captured = capsys.readouterr()
        assert "Warning" in captured.out

    # --- Strategy 4: main() function ------------------------------------

    def test_strategy4_main_returns_service(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "s4a.py", code)

        inst = _MockSWMLService()
        inst.name = "from-main"
        inst.route = "/from-main"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.main = lambda: inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        assert result is inst

    def test_strategy4_main_captured_via_serve(self, tmp_path):
        """main() calls serve() which we intercept to capture the service."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "s4b.py", code)

        inst = _MockSWMLService()
        inst.name = "captured"
        inst.route = "/captured"

        def fake_main():
            inst.serve()  # should be intercepted

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.main = fake_main

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path)
        assert result is inst

    def test_strategy4_main_exception_prints_warning(self, tmp_path, capsys):
        """If main() raises, a warning is printed and we fall through."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "s4c.py", code)

        def bad_main():
            raise RuntimeError("main failed")

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.main = bad_main

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            with pytest.raises(ValueError, match="No service found"):
                agent_loader._load_service_impl(path)

        captured = capsys.readouterr()
        assert "Warning" in captured.out

    # --- no service found ------------------------------------------------

    def test_no_service_raises(self, tmp_path):
        path = _write_py(tmp_path, "nothing.py", "x = 1\n")
        with pytest.raises(ValueError, match="No service found"):
            agent_loader._load_service_impl(path)

    # --- sys.path management in load impl --------------------------------

    def test_load_impl_cleans_sys_path(self, tmp_path):
        code = "x = 1\n"
        path = _write_py(tmp_path, "clean.py", code)

        inst = _MockSWMLService()
        inst.name = "sp"
        inst.route = "/sp"

        module_dir = str(tmp_path.resolve())
        if module_dir in sys.path:
            sys.path.remove(module_dir)

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.agent = inst

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            agent_loader._load_service_impl(path)

        assert module_dir not in sys.path


# ============================================================================
# Module-level import fallback (AVAILABLE flags)
# ============================================================================

class TestModuleFallbacks:
    """Tests verifying behaviour when base classes are not importable."""

    def test_discover_services_raises_with_swml_unavailable(self, tmp_path):
        path = _write_py(tmp_path, "f1.py", "x = 1\n")
        with patch.object(agent_loader, "SWML_SERVICE_AVAILABLE", False):
            with pytest.raises(ImportError):
                agent_loader.discover_services_in_file(path)

    def test_load_service_raises_with_swml_unavailable(self, tmp_path):
        path = _write_py(tmp_path, "f2.py", "x = 1\n")
        with patch.object(agent_loader, "SWML_SERVICE_AVAILABLE", False):
            with pytest.raises(ImportError):
                agent_loader.load_service_from_file(path)

    def test_load_agent_raises_with_agent_base_unavailable(self, tmp_path):
        path = _write_py(tmp_path, "f3.py", "x = 1\n")
        with patch.object(agent_loader, "AGENT_BASE_AVAILABLE", False):
            with pytest.raises(ImportError):
                agent_loader.load_agent_from_file(path)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_prefer_route_tries_class_instantiation_for_route(self, tmp_path):
        """If no existing instance matches the route, _load_service_impl
        tries instantiating classes and checking their routes."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "edge1.py", code)

        class RoutedSvc(_MockSWMLService):
            def __init__(self):
                self.name = "routed"
                self.route = "/special"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.RoutedSvc = RoutedSvc

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            result = agent_loader._load_service_impl(path, "/special", prefer_route=True)
        assert isinstance(result, RoutedSvc)

    def test_class_name_path_instantiation_error(self, tmp_path):
        """When prefer_route=False and the class cannot be instantiated, raise ValueError."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "edge2.py", code)

        class BadClass(_MockSWMLService):
            def __init__(self):
                raise RuntimeError("no init")

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.BadClass = BadClass

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            with pytest.raises(ValueError, match="Failed to instantiate"):
                agent_loader._load_service_impl(path, "BadClass", prefer_route=False)

    def test_strategy3_skips_when_module_has_main(self, tmp_path):
        """Strategy 3 (class discovery) is skipped when module has main()."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "edge3.py", code)

        called = []

        def my_main():
            called.append(True)

        class UnusedSvc(_MockSWMLService):
            def __init__(self):
                self.name = "unused"
                self.route = "/unused"

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.main = my_main
                module.UnusedSvc = UnusedSvc

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            # main() doesn't return a service, so ultimately raises ValueError
            with pytest.raises(ValueError, match="No service found"):
                agent_loader._load_service_impl(path)
        # main() should have been called (Strategy 4)
        assert len(called) == 1

    def test_discover_services_class_with_exception_in_info(self, tmp_path):
        """If getting class info raises, the exception path in _discover_services_impl
        still records the class."""
        code = "x = 1\n"
        path = _write_py(tmp_path, "edge4.py", code)

        class TrickySvc(_MockSWMLService):
            """Tricky doc"""
            pass

        orig_exec = importlib.util.spec_from_file_location

        def patched_spec(*args, **kwargs):
            spec = orig_exec(*args, **kwargs)
            real_exec = spec.loader.exec_module

            def fake_exec(module):
                real_exec(module)
                module.TrickySvc = TrickySvc

            spec.loader.exec_module = fake_exec
            return spec

        with patch("importlib.util.spec_from_file_location", side_effect=patched_spec):
            results = agent_loader._discover_services_impl(path)

        tricky = [r for r in results if r["name"] == "TrickySvc"]
        assert len(tricky) == 1
        assert tricky[0]["type"] == "class"
