"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for ToolMixin
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from signalwire_agents.core.mixins.tool_mixin import ToolMixin
from signalwire_agents.core.function_result import SwaigFunctionResult
from signalwire_agents.core.swaig_function import SWAIGFunction


class MockToolHost(ToolMixin):
    """
    A minimal host class that inherits from ToolMixin and provides
    all the attributes the mixin expects to find on self.
    """

    def __init__(self, tool_registry=None):
        self._tool_registry = tool_registry or Mock()
        self.log = Mock()
        # Bind returns another mock logger
        self.log.bind.return_value = self.log


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_registry():
    """Return a fresh Mock for ToolRegistry."""
    reg = Mock()
    reg._swaig_functions = {}
    return reg


@pytest.fixture
def host(mock_registry):
    """Return a MockToolHost wired with a mock tool registry."""
    return MockToolHost(tool_registry=mock_registry)


def _make_swaig_function(name="test_tool", handler=None, secure=True, webhook_url=None):
    """Helper to create a SWAIGFunction instance."""
    if handler is None:
        handler = Mock(return_value=SwaigFunctionResult("ok"))
    return SWAIGFunction(
        name=name,
        handler=handler,
        description=f"Description for {name}",
        parameters={"input": {"type": "string", "description": "An input"}},
        secure=secure,
        webhook_url=webhook_url,
    )


# ===========================================================================
# Tests for define_tool
# ===========================================================================

class TestDefineTool:
    """Tests for ToolMixin.define_tool"""

    def test_delegates_to_registry(self, host, mock_registry):
        handler = Mock()
        host.define_tool(
            name="my_func",
            description="Does stuff",
            parameters={"x": {"type": "string"}},
            handler=handler,
        )
        mock_registry.define_tool.assert_called_once_with(
            name="my_func",
            description="Does stuff",
            parameters={"x": {"type": "string"}},
            handler=handler,
            secure=True,
            fillers=None,
            webhook_url=None,
            required=None,
            is_typed_handler=False,
        )

    def test_returns_self_for_chaining(self, host):
        result = host.define_tool(
            name="f", description="d", parameters={}, handler=Mock()
        )
        assert result is host

    def test_passes_secure_false(self, host, mock_registry):
        host.define_tool(
            name="f", description="d", parameters={}, handler=Mock(), secure=False
        )
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["secure"] is False

    def test_passes_fillers(self, host, mock_registry):
        fillers = {"en": ["one moment", "please wait"]}
        host.define_tool(
            name="f", description="d", parameters={}, handler=Mock(), fillers=fillers
        )
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["fillers"] == fillers

    def test_passes_webhook_url(self, host, mock_registry):
        host.define_tool(
            name="f", description="d", parameters={}, handler=Mock(),
            webhook_url="https://example.com/hook"
        )
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["webhook_url"] == "https://example.com/hook"

    def test_passes_required(self, host, mock_registry):
        host.define_tool(
            name="f", description="d", parameters={}, handler=Mock(),
            required=["x", "y"]
        )
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["required"] == ["x", "y"]

    def test_chain_multiple_define_tool_calls(self, host):
        result = (
            host
            .define_tool(name="a", description="A", parameters={}, handler=Mock())
            .define_tool(name="b", description="B", parameters={}, handler=Mock())
        )
        assert result is host

    def test_passes_is_typed_handler(self, host, mock_registry):
        host.define_tool(
            name="f", description="d", parameters={}, handler=Mock(),
            is_typed_handler=True
        )
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["is_typed_handler"] is True

    def test_is_typed_handler_defaults_false(self, host, mock_registry):
        host.define_tool(
            name="f", description="d", parameters={}, handler=Mock()
        )
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["is_typed_handler"] is False


# ===========================================================================
# Tests for register_swaig_function
# ===========================================================================

class TestRegisterSwaigFunction:
    """Tests for ToolMixin.register_swaig_function"""

    def test_delegates_to_registry(self, host, mock_registry):
        func_dict = {"function": "data_map_func", "data_map": {"url": "https://example.com"}}
        host.register_swaig_function(func_dict)
        mock_registry.register_swaig_function.assert_called_once_with(func_dict)

    def test_returns_self_for_chaining(self, host):
        result = host.register_swaig_function({"function": "f", "data_map": {}})
        assert result is host


# ===========================================================================
# Tests for define_tools
# ===========================================================================

class TestDefineTools:
    """Tests for ToolMixin.define_tools"""

    def test_returns_empty_list_when_no_functions(self, host, mock_registry):
        mock_registry._swaig_functions = {}
        result = host.define_tools()
        assert result == []

    def test_returns_swaig_function_objects(self, host, mock_registry):
        func = _make_swaig_function("tool1")
        mock_registry._swaig_functions = {"tool1": func}
        result = host.define_tools()
        assert len(result) == 1
        assert result[0] is func

    def test_returns_raw_dicts_for_data_map(self, host, mock_registry):
        data_map = {"function": "dm_func", "data_map": {"url": "https://example.com"}}
        mock_registry._swaig_functions = {"dm_func": data_map}
        result = host.define_tools()
        assert len(result) == 1
        assert result[0] is data_map

    def test_returns_mixed_types(self, host, mock_registry):
        func = _make_swaig_function("regular")
        data_map = {"function": "dm", "data_map": {}}
        mock_registry._swaig_functions = {"regular": func, "dm": data_map}
        result = host.define_tools()
        assert len(result) == 2


# ===========================================================================
# Tests for on_function_call
# ===========================================================================

class TestOnFunctionCall:
    """Tests for ToolMixin.on_function_call"""

    def test_unknown_function_returns_error(self, host, mock_registry):
        mock_registry._swaig_functions = {}
        result = host.on_function_call("nonexistent", {})
        assert "not found" in result["response"]

    def test_data_map_function_returns_error(self, host, mock_registry):
        mock_registry._swaig_functions = {
            "dm": {"function": "dm", "data_map": {}}
        }
        result = host.on_function_call("dm", {})
        assert "Data map" in result["response"]

    def test_webhook_function_returns_error(self, host, mock_registry):
        func = _make_swaig_function("webhook_func", webhook_url="https://example.com")
        mock_registry._swaig_functions = {"webhook_func": func}
        result = host.on_function_call("webhook_func", {})
        assert "webhook" in result["response"].lower() or "External" in result["response"]

    def test_calls_handler_successfully(self, host, mock_registry):
        expected_result = SwaigFunctionResult("success")
        handler = Mock(return_value=expected_result)
        func = _make_swaig_function("my_tool", handler=handler)
        mock_registry._swaig_functions = {"my_tool": func}

        result = host.on_function_call("my_tool", {"key": "val"}, {"raw": "data"})
        handler.assert_called_once_with({"key": "val"}, {"raw": "data"})
        assert result is expected_result

    def test_handler_returning_none_creates_default(self, host, mock_registry):
        handler = Mock(return_value=None)
        func = _make_swaig_function("my_tool", handler=handler)
        mock_registry._swaig_functions = {"my_tool": func}

        result = host.on_function_call("my_tool", {})
        assert isinstance(result, SwaigFunctionResult)

    def test_handler_exception_returns_error(self, host, mock_registry):
        handler = Mock(side_effect=RuntimeError("handler crash"))
        func = _make_swaig_function("my_tool", handler=handler)
        mock_registry._swaig_functions = {"my_tool": func}

        result = host.on_function_call("my_tool", {})
        assert "Error" in result["response"]


# ===========================================================================
# Tests for _execute_swaig_function
# ===========================================================================

class TestExecuteSwaigFunction:
    """Tests for ToolMixin._execute_swaig_function"""

    def test_function_not_found(self, host, mock_registry):
        mock_registry._swaig_functions = {}
        result = host._execute_swaig_function("nonexistent")
        assert "error" in result

    def test_default_args_when_none(self, host, mock_registry):
        handler = Mock(return_value=SwaigFunctionResult("done"))
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        result = host._execute_swaig_function("tool")
        assert "response" in result
        assert result["response"] == "done"

    def test_passes_args_and_raw_data(self, host, mock_registry):
        handler = Mock(return_value=SwaigFunctionResult("ok"))
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        raw = {"function": "tool", "call_id": "c1"}
        result = host._execute_swaig_function("tool", args={"key": "val"}, raw_data=raw)
        handler.assert_called_once()
        assert result["response"] == "ok"

    def test_with_call_id(self, host, mock_registry):
        handler = Mock(return_value=SwaigFunctionResult("ok"))
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        result = host._execute_swaig_function("tool", args={}, call_id="call-42")
        assert result["response"] == "ok"

    def test_constructs_raw_data_with_args(self, host, mock_registry):
        handler = Mock(return_value=SwaigFunctionResult("fine"))
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        host._execute_swaig_function("tool", args={"a": 1}, call_id="c1")
        # Verify handler was called with constructed raw_data
        call_args = handler.call_args
        raw_data = call_args[0][1]
        assert raw_data["function"] == "tool"
        assert raw_data["call_id"] == "c1"

    def test_handler_returning_dict(self, host, mock_registry):
        handler = Mock(return_value={"response": "dict result"})
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        result = host._execute_swaig_function("tool")
        assert result["response"] == "dict result"

    def test_handler_returning_string(self, host, mock_registry):
        handler = Mock(return_value="just a string")
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        result = host._execute_swaig_function("tool")
        assert "response" in result

    def test_handler_exception_returns_error_response(self, host, mock_registry):
        handler = Mock(side_effect=RuntimeError("boom"))
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        result = host._execute_swaig_function("tool")
        # on_function_call catches the exception and returns {"response": "Error..."}
        # which _execute_swaig_function passes through as a dict result
        assert "response" in result
        assert "Error" in result["response"]

    def test_empty_args_creates_empty_raw_data_argument(self, host, mock_registry):
        handler = Mock(return_value=SwaigFunctionResult("ok"))
        func = _make_swaig_function("tool", handler=handler)
        mock_registry._swaig_functions = {"tool": func}

        host._execute_swaig_function("tool", args=None)
        call_args = handler.call_args
        raw_data = call_args[0][1]
        assert raw_data["argument"]["raw"] == "{}"


# ===========================================================================
# Tests for _tool_decorator
# ===========================================================================

class TestToolDecorator:
    """Tests for ToolMixin._tool_decorator"""

    def test_decorator_returns_callable(self, host):
        decorator = host._tool_decorator(name="test_func")
        assert callable(decorator)

    def test_decorated_function_is_registered(self, host, mock_registry):
        mock_registry.define_tool = Mock()

        @host._tool_decorator(name="greet", description="Greet user", parameters={"name": {"type": "string"}})
        def greet(args, raw_data):
            return SwaigFunctionResult("Hello")

        mock_registry.define_tool.assert_called_once()
        call_kwargs = mock_registry.define_tool.call_args[1]
        assert call_kwargs["name"] == "greet"
        assert call_kwargs["description"] == "Greet user"


# ===========================================================================
# Tests for tool class decorator
# ===========================================================================

class TestToolClassDecorator:
    """Tests for ToolMixin.tool class method decorator"""

    def test_class_decorator_marks_function(self):
        decorator = ToolMixin.tool(name="class_func", parameters={})

        def my_func(self, args, raw_data):
            return SwaigFunctionResult("hi")

        decorated = decorator(my_func)
        assert decorated._is_tool is True
        assert decorated._tool_name == "class_func"

    def test_class_decorator_uses_function_name_when_no_name(self):
        decorator = ToolMixin.tool(parameters={})

        def my_func(self, args, raw_data):
            pass

        decorated = decorator(my_func)
        assert decorated._tool_name == "my_func"

    def test_class_decorator_preserves_function(self):
        decorator = ToolMixin.tool(name="func")

        def my_func(self, args, raw_data):
            return "original"

        decorated = decorator(my_func)
        # The decorated function should still be the original
        assert decorated is my_func
