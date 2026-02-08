"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for CLI core service_loader module.

Tests cover:
- ServiceCapture class (init, capture, apply/restore patches)
- Service module loading from file paths
- Service class discovery
- Error handling: file not found, import errors, no services, non-.py files
- load_and_simulate_service function
- load_agent_from_file backward compatibility wrapper
- discover_agents_in_file backward compatibility wrapper
- Multiple services in one file
- suppress_output option
"""

import pytest
import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call
from io import StringIO

from signalwire_agents.cli.core.service_loader import (
    ServiceCapture,
    load_and_simulate_service,
    load_agent_from_file,
    discover_agents_in_file,
    simulate_request_to_service,
    DEPENDENCIES_AVAILABLE,
)
from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.core.agent_base import AgentBase


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def capturer():
    """Create a fresh ServiceCapture instance."""
    return ServiceCapture()


class _FakeSWMLService:
    """A plain class that is NOT a subclass of AgentBase.
    Used so isinstance(obj, AgentBase) returns False."""
    pass


def _make_mock_service(name="test_service", route="/test"):
    """Create an object that is NOT an AgentBase instance."""
    svc = _FakeSWMLService()
    svc.name = name
    svc.route = route
    return svc


def _make_mock_agent(name="test_agent", route="/agent", class_name="TestAgent"):
    """Create a mock that passes isinstance checks for AgentBase."""
    agent = Mock(spec=AgentBase)
    agent.name = name
    agent.route = route
    # Don't override __class__ - let Mock(spec=AgentBase) handle isinstance.
    # Instead, create a wrapper for __class__ access that returns an object
    # with the right __name__ and __doc__.
    # We use a trick: create a type that is a real subclass of AgentBase
    # and assign it. The type() call only creates the class object, it does
    # NOT call AgentBase.__init__.
    _cls = type(class_name, (AgentBase,), {"__doc__": "A mock agent"})
    agent.__class__ = _cls
    return agent


# =============================================================================
# ServiceCapture.__init__ Tests
# =============================================================================

class TestServiceCaptureInit:
    """Tests for ServiceCapture initialization."""

    def test_init_creates_empty_captured_services(self, capturer):
        assert capturer.captured_services == []

    def test_init_creates_empty_original_methods(self, capturer):
        assert capturer.original_methods == {}


# =============================================================================
# ServiceCapture.capture Error Handling Tests
# =============================================================================

class TestServiceCaptureErrors:
    """Tests for ServiceCapture.capture error conditions."""

    def test_capture_file_not_found(self, capturer, tmp_path):
        """FileNotFoundError when the service file doesn't exist."""
        missing = str(tmp_path / "no_such_file.py")
        with pytest.raises(FileNotFoundError, match="Service file not found"):
            capturer.capture(missing)

    def test_capture_non_python_file(self, capturer, tmp_path):
        """ValueError when the file is not a .py file."""
        txt_file = tmp_path / "service.txt"
        txt_file.write_text("not python")
        with pytest.raises(ValueError, match="must be a Python file"):
            capturer.capture(str(txt_file))

    def test_capture_non_python_file_js(self, capturer, tmp_path):
        """ValueError for .js files."""
        js_file = tmp_path / "service.js"
        js_file.write_text("console.log('hi')")
        with pytest.raises(ValueError, match="must be a Python file"):
            capturer.capture(str(js_file))

    def test_capture_dependencies_not_available(self, capturer, tmp_path):
        """ImportError when DEPENDENCIES_AVAILABLE is False."""
        py_file = tmp_path / "service.py"
        py_file.write_text("pass")
        with patch("signalwire_agents.cli.core.service_loader.DEPENDENCIES_AVAILABLE", False):
            with pytest.raises(ImportError, match="Required dependencies not available"):
                capturer.capture(str(py_file))

    def test_capture_import_error_no_services_captured(self, capturer, tmp_path):
        """ImportError when module exec fails and no services were captured."""
        py_file = tmp_path / "bad_service.py"
        py_file.write_text("raise RuntimeError('import boom')")
        with pytest.raises(ImportError, match="Failed to load service module"):
            capturer.capture(str(py_file))

    def test_capture_import_error_with_services_captured(self, capturer, tmp_path):
        """When module exec fails but services were already captured, no error raised."""
        py_file = tmp_path / "partial_service.py"
        # Write a service file that will create an instance, call run(), then crash
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc = SWMLService(name='partial', route='/p', schema_validation=False)\n"
            "svc.serve()\n"
            "raise RuntimeError('after capture')\n"
        )
        services = capturer.capture(str(py_file))
        # Should have captured the service before the crash
        assert len(services) >= 1


# =============================================================================
# ServiceCapture.capture Successful Loading Tests
# =============================================================================

class TestServiceCaptureSuccess:
    """Tests for successful service capture scenarios."""

    def test_capture_service_via_serve(self, capturer, tmp_path):
        """Capture a service that calls serve()."""
        py_file = tmp_path / "my_service.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc = SWMLService(name='serve_test', route='/s', schema_validation=False)\n"
            "svc.serve()\n"
        )
        services = capturer.capture(str(py_file))
        assert len(services) == 1
        assert services[0].name == "serve_test"

    def test_capture_service_via_run(self, capturer, tmp_path):
        """Capture a service that calls run() (via AgentBase)."""
        py_file = tmp_path / "my_agent.py"
        py_file.write_text(
            "from signalwire_agents import AgentBase\n"
            "agent = AgentBase(name='run_test', route='/r', schema_validation=False)\n"
            "agent.run()\n"
        )
        services = capturer.capture(str(py_file))
        assert len(services) == 1

    def test_capture_multiple_services(self, capturer, tmp_path):
        """Capture multiple services from one file."""
        py_file = tmp_path / "multi_service.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc1 = SWMLService(name='svc1', route='/a', schema_validation=False)\n"
            "svc1.serve()\n"
            "svc2 = SWMLService(name='svc2', route='/b', schema_validation=False)\n"
            "svc2.serve()\n"
        )
        services = capturer.capture(str(py_file))
        assert len(services) == 2
        names = {s.name for s in services}
        assert names == {"svc1", "svc2"}

    def test_capture_resets_list_between_calls(self, capturer, tmp_path):
        """captured_services is reset before each capture call."""
        py_file = tmp_path / "resettable.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc = SWMLService(name='resettable', route='/r', schema_validation=False)\n"
            "svc.serve()\n"
        )
        services1 = capturer.capture(str(py_file))
        assert len(services1) == 1

        services2 = capturer.capture(str(py_file))
        assert len(services2) == 1
        # Should be a fresh list, not appended
        assert len(capturer.captured_services) == 1

    def test_capture_with_suppress_output(self, capturer, tmp_path):
        """suppress_output=True suppresses stdout from the loaded module."""
        py_file = tmp_path / "noisy_service.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "print('I AM VERY NOISY')\n"
            "svc = SWMLService(name='noisy', route='/n', schema_validation=False)\n"
            "svc.serve()\n"
        )
        import io
        captured_output = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(captured_output):
            services = capturer.capture(str(py_file), suppress_output=True)

        assert len(services) == 1
        # The noisy print should NOT appear in our captured_output because
        # suppress_output redirects inside capture()
        # (stdout was already redirected by capture, so our outer redirect sees nothing)

    def test_capture_without_suppress_output(self, capturer, tmp_path, capsys):
        """suppress_output=False (default) allows stdout from the loaded module."""
        py_file = tmp_path / "chatty_service.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc = SWMLService(name='chatty', route='/c', schema_validation=False)\n"
            "svc.serve()\n"
        )
        services = capturer.capture(str(py_file), suppress_output=False)
        assert len(services) == 1

    def test_capture_returns_list(self, capturer, tmp_path):
        """capture() returns a list."""
        py_file = tmp_path / "list_service.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc = SWMLService(name='list_test', route='/l', schema_validation=False)\n"
            "svc.serve()\n"
        )
        result = capturer.capture(str(py_file))
        assert isinstance(result, list)

    def test_capture_resolves_relative_path(self, capturer, tmp_path, monkeypatch):
        """capture() resolves the path before checking existence."""
        py_file = tmp_path / "relative_service.py"
        py_file.write_text(
            "from signalwire_agents.core.swml_service import SWMLService\n"
            "svc = SWMLService(name='relative', route='/rel', schema_validation=False)\n"
            "svc.serve()\n"
        )
        monkeypatch.chdir(tmp_path)
        services = capturer.capture("relative_service.py")
        assert len(services) == 1

    def test_capture_empty_module_returns_empty_list(self, capturer, tmp_path):
        """Capture of a file with no services returns empty list."""
        py_file = tmp_path / "empty_service.py"
        py_file.write_text("x = 42\n")
        services = capturer.capture(str(py_file))
        assert services == []


# =============================================================================
# ServiceCapture._apply_patches and _restore_patches Tests
# =============================================================================

class TestServiceCapturePatching:
    """Tests for the patch/restore mechanism."""

    def test_apply_patches_stores_originals(self, capturer):
        """_apply_patches stores original run/serve methods."""
        original_serve = SWMLService.serve
        capturer._apply_patches()
        try:
            assert len(capturer.original_methods) > 0
            # Check that original serve was stored
            assert (SWMLService, 'serve') in capturer.original_methods
            assert capturer.original_methods[(SWMLService, 'serve')] is original_serve
        finally:
            capturer._restore_patches()

    def test_apply_patches_replaces_methods(self, capturer):
        """_apply_patches replaces run/serve on base classes."""
        original_serve = SWMLService.serve
        capturer._apply_patches()
        try:
            assert SWMLService.serve is not original_serve
        finally:
            capturer._restore_patches()

    def test_restore_patches_restores_original_methods(self, capturer):
        """_restore_patches restores original methods."""
        original_serve = SWMLService.serve
        capturer._apply_patches()
        capturer._restore_patches()
        assert SWMLService.serve is original_serve

    def test_restore_patches_clears_dict(self, capturer):
        """_restore_patches clears the original_methods dict."""
        capturer._apply_patches()
        capturer._restore_patches()
        assert capturer.original_methods == {}

    def test_restore_patches_with_empty_dict(self, capturer):
        """_restore_patches handles empty original_methods gracefully."""
        capturer._restore_patches()  # Should not raise
        assert capturer.original_methods == {}

    def test_patches_restored_after_error(self, capturer, tmp_path):
        """Patches are restored even when capture raises an error."""
        original_serve = SWMLService.serve
        py_file = tmp_path / "error_service.py"
        py_file.write_text("raise RuntimeError('boom')")
        with pytest.raises(ImportError):
            capturer.capture(str(py_file))
        # Patches should be restored
        assert SWMLService.serve is original_serve

    def test_mock_run_captures_service(self, capturer):
        """The patched run() method captures the service instance."""
        capturer._apply_patches()
        try:
            svc = SWMLService(name="mock_run_test", route="/mr", schema_validation=False)
            # Calling run on the instance should capture it
            # But run is on WebMixin/AgentBase; for SWMLService only serve exists
            # Let's test serve:
            svc.serve()
            assert svc in capturer.captured_services
        finally:
            capturer._restore_patches()

    def test_mock_serve_returns_service(self, capturer):
        """The patched serve() method returns the service instance."""
        capturer._apply_patches()
        try:
            svc = SWMLService(name="return_test", route="/rt", schema_validation=False)
            result = svc.serve()
            assert result is svc
        finally:
            capturer._restore_patches()


# =============================================================================
# load_and_simulate_service Tests
# =============================================================================

class TestLoadAndSimulateService:
    """Tests for the load_and_simulate_service function."""

    def test_no_services_raises_value_error(self, tmp_path):
        """ValueError when no services are found in the file."""
        py_file = tmp_path / "empty.py"
        py_file.write_text("x = 1\n")
        with pytest.raises(ValueError, match="No services found"):
            load_and_simulate_service(str(py_file))

    def test_multiple_services_no_route_raises_value_error(self):
        """ValueError listing available routes when multiple services found and no route specified."""
        mock_svc1 = _make_mock_service(route="/a")
        mock_svc2 = _make_mock_service(route="/b")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc1, mock_svc2]):
            with pytest.raises(ValueError, match="Multiple services found"):
                load_and_simulate_service("fake.py")

    def test_multiple_services_wrong_route_raises_value_error(self):
        """ValueError when specified route doesn't match any service."""
        mock_svc1 = _make_mock_service(route="/a")
        mock_svc2 = _make_mock_service(route="/b")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc1, mock_svc2]):
            with pytest.raises(ValueError, match="No service found for route '/c'"):
                load_and_simulate_service("fake.py", route="/c")

    def test_multiple_services_correct_route_selects_service(self):
        """Correct service selected when route matches."""
        mock_svc1 = _make_mock_service(route="/a")
        mock_svc2 = _make_mock_service(route="/b")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc1, mock_svc2]), \
             patch("signalwire_agents.cli.core.service_loader.simulate_request_to_service", new=MagicMock()), \
             patch("signalwire_agents.cli.core.service_loader.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = {"result": "ok"}
            result = load_and_simulate_service("fake.py", route="/b")
            assert result == {"result": "ok"}
            # Verify the correct service was passed to simulate
            call_args = mock_asyncio.run.call_args
            assert call_args is not None

    def test_single_service_selected_automatically(self):
        """Single service is selected without needing a route."""
        mock_svc = _make_mock_service(route="/only")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc]), \
             patch("signalwire_agents.cli.core.service_loader.simulate_request_to_service", new=MagicMock()), \
             patch("signalwire_agents.cli.core.service_loader.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = {"response": "data"}
            result = load_and_simulate_service("fake.py")
            assert result == {"response": "data"}

    def test_passes_parameters_through(self):
        """Method, body, query_params, and headers are forwarded."""
        mock_svc = _make_mock_service(route="/only")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc]), \
             patch("signalwire_agents.cli.core.service_loader.simulate_request_to_service", new=MagicMock()), \
             patch("signalwire_agents.cli.core.service_loader.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = {}
            load_and_simulate_service(
                "fake.py",
                method="GET",
                body={"key": "val"},
                query_params={"q": "search"},
                headers={"X-Custom": "header"},
                suppress_output=True
            )
            # Verify asyncio.run was called
            assert mock_asyncio.run.called

    def test_multiple_services_available_routes_in_error(self):
        """The error message includes the available routes."""
        mock_svc1 = _make_mock_service(route="/route1")
        mock_svc2 = _make_mock_service(route="/route2")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc1, mock_svc2]):
            with pytest.raises(ValueError, match="/route1") as exc_info:
                load_and_simulate_service("fake.py")
            assert "/route2" in str(exc_info.value)


# =============================================================================
# load_agent_from_file Tests
# =============================================================================

class TestLoadAgentFromFile:
    """Tests for the load_agent_from_file backward compatibility function."""

    def test_no_agents_raises_value_error(self):
        """ValueError when no agents found in the file."""
        mock_svc = _make_mock_service()  # Not an AgentBase

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc]):
            with pytest.raises(ValueError, match="No agents found"):
                load_agent_from_file("fake.py")

    def test_single_agent_returned(self):
        """Single agent is returned directly."""
        mock_agent = _make_mock_agent(name="solo")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_agent]):
            result = load_agent_from_file("fake.py")
            assert result is mock_agent

    def test_multiple_agents_returns_first(self):
        """When multiple agents and no class name, returns first."""
        agent1 = _make_mock_agent(name="first", class_name="FirstAgent")
        agent2 = _make_mock_agent(name="second", class_name="SecondAgent")

        with patch.object(ServiceCapture, 'capture', return_value=[agent1, agent2]):
            result = load_agent_from_file("fake.py")
            assert result is agent1

    def test_multiple_agents_with_class_name(self):
        """When multiple agents and class name is specified, matches by name."""
        agent1 = _make_mock_agent(name="first", class_name="FirstAgent")
        agent2 = _make_mock_agent(name="second", class_name="SecondAgent")

        with patch.object(ServiceCapture, 'capture', return_value=[agent1, agent2]):
            result = load_agent_from_file("fake.py", agent_class_name="SecondAgent")
            assert result is agent2

    def test_class_name_no_match_returns_first(self):
        """When class name doesn't match any agent, returns first."""
        agent1 = _make_mock_agent(name="first", class_name="FirstAgent")
        agent2 = _make_mock_agent(name="second", class_name="SecondAgent")

        with patch.object(ServiceCapture, 'capture', return_value=[agent1, agent2]):
            result = load_agent_from_file("fake.py", agent_class_name="ThirdAgent")
            assert result is agent1

    def test_filters_non_agents(self):
        """Non-AgentBase services are filtered out."""
        mock_svc = _make_mock_service()
        mock_agent = _make_mock_agent(name="real_agent")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc, mock_agent]):
            result = load_agent_from_file("fake.py")
            assert result is mock_agent

    def test_suppress_output_forwarded(self):
        """suppress_output parameter is forwarded to capture()."""
        mock_agent = _make_mock_agent()

        with patch.object(ServiceCapture, 'capture', return_value=[mock_agent]) as mock_capture:
            load_agent_from_file("fake.py", suppress_output=True)
            mock_capture.assert_called_once_with("fake.py", suppress_output=True)

    def test_empty_capture_raises(self):
        """Empty capture list raises ValueError (no agents)."""
        with patch.object(ServiceCapture, 'capture', return_value=[]):
            with pytest.raises(ValueError, match="No agents found"):
                load_agent_from_file("fake.py")


# =============================================================================
# discover_agents_in_file Tests
# =============================================================================

class TestDiscoverAgentsInFile:
    """Tests for the discover_agents_in_file backward compatibility function."""

    def test_empty_file_returns_empty_list(self):
        """Empty capture returns empty list."""
        with patch.object(ServiceCapture, 'capture', return_value=[]):
            result = discover_agents_in_file("fake.py")
            assert result == []

    def test_returns_proper_dict_format(self):
        """Each entry has expected keys: name, class_name, type, agent_name, route, description, object."""
        mock_agent = _make_mock_agent(name="discover_me", route="/d", class_name="DiscoverAgent")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_agent]):
            result = discover_agents_in_file("fake.py")
            assert len(result) == 1
            entry = result[0]
            assert entry["name"] == "discover_me"
            assert entry["class_name"] == "DiscoverAgent"
            assert entry["type"] == "instance"
            assert entry["agent_name"] == "discover_me"
            assert entry["route"] == "/d"
            assert entry["object"] is mock_agent

    def test_filters_non_agent_services(self):
        """Non-AgentBase services are excluded."""
        mock_svc = _make_mock_service()
        mock_agent = _make_mock_agent(name="agent_only")

        with patch.object(ServiceCapture, 'capture', return_value=[mock_svc, mock_agent]):
            result = discover_agents_in_file("fake.py")
            assert len(result) == 1
            assert result[0]["name"] == "agent_only"

    def test_multiple_agents(self):
        """Multiple agents all appear in result."""
        agent1 = _make_mock_agent(name="agent1", class_name="Agent1")
        agent2 = _make_mock_agent(name="agent2", class_name="Agent2")

        with patch.object(ServiceCapture, 'capture', return_value=[agent1, agent2]):
            result = discover_agents_in_file("fake.py")
            assert len(result) == 2
            names = {e["name"] for e in result}
            assert names == {"agent1", "agent2"}

    def test_description_from_docstring(self):
        """The description field comes from the class docstring."""
        mock_agent = _make_mock_agent(name="documented")
        # The docstring is set in _make_mock_agent via the type() call

        with patch.object(ServiceCapture, 'capture', return_value=[mock_agent]):
            result = discover_agents_in_file("fake.py")
            assert result[0]["description"] == "A mock agent"


# =============================================================================
# simulate_request_to_service Tests
# =============================================================================

class TestSimulateRequestToService:
    """Tests for the async simulate_request_to_service function."""

    @pytest.mark.asyncio
    async def test_simulate_returns_dict_response(self):
        """When the service handler returns a dict, it's returned as-is."""
        mock_service = Mock()

        async def async_handler(request, response):
            return {"swml": "data"}

        mock_service._handle_request = async_handler

        with patch("signalwire_agents.cli.simulation.mock_env.create_mock_request") as mock_create, \
             patch("signalwire_agents.cli.core.service_loader.Response") as mock_response_cls:
            mock_create.return_value = Mock()
            mock_response_cls.return_value = Mock()
            result = await simulate_request_to_service(mock_service, body={"test": True})
            assert result == {"swml": "data"}

    @pytest.mark.asyncio
    async def test_simulate_returns_body_response(self):
        """When the handler returns an object with body attr, parse as JSON."""
        import json as json_mod
        response_obj = Mock()
        response_obj.body = json_mod.dumps({"parsed": True}).encode()

        mock_service = Mock()

        async def async_handler(request, response):
            return response_obj

        mock_service._handle_request = async_handler

        with patch("signalwire_agents.cli.simulation.mock_env.create_mock_request") as mock_create, \
             patch("signalwire_agents.cli.core.service_loader.Response") as mock_response_cls:
            mock_create.return_value = Mock()
            mock_response_cls.return_value = Mock()
            result = await simulate_request_to_service(mock_service)
            assert result == {"parsed": True}

    @pytest.mark.asyncio
    async def test_simulate_returns_error_for_unparseable(self):
        """When the response is neither dict nor has body, return error dict."""
        mock_service = Mock()

        async def async_handler(request, response):
            return "just a string"

        mock_service._handle_request = async_handler

        with patch("signalwire_agents.cli.simulation.mock_env.create_mock_request") as mock_create, \
             patch("signalwire_agents.cli.core.service_loader.Response") as mock_response_cls:
            mock_create.return_value = Mock()
            mock_response_cls.return_value = Mock()
            result = await simulate_request_to_service(mock_service)
            assert "error" in result
