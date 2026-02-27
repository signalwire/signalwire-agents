"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Comprehensive tests for type-hint-based tool schema inference.
"""

import pytest
from typing import Optional, List, Dict, Literal
from unittest.mock import Mock, MagicMock

from signalwire_agents.core.agent.tools.type_inference import (
    infer_schema,
    create_typed_handler_wrapper,
    _resolve_type,
    _parse_docstring_args,
)
from signalwire_agents.core.function_result import SwaigFunctionResult


# ===========================================================================
# Tests for _resolve_type
# ===========================================================================

class TestResolveType:
    """Tests for the _resolve_type helper."""

    def test_str(self):
        schema, optional = _resolve_type(str)
        assert schema == {"type": "string"}
        assert optional is False

    def test_int(self):
        schema, optional = _resolve_type(int)
        assert schema == {"type": "integer"}
        assert optional is False

    def test_float(self):
        schema, optional = _resolve_type(float)
        assert schema == {"type": "number"}
        assert optional is False

    def test_bool(self):
        schema, optional = _resolve_type(bool)
        assert schema == {"type": "boolean"}
        assert optional is False

    def test_list(self):
        schema, optional = _resolve_type(list)
        assert schema == {"type": "array"}
        assert optional is False

    def test_dict(self):
        schema, optional = _resolve_type(dict)
        assert schema == {"type": "object"}
        assert optional is False

    def test_optional_str(self):
        schema, optional = _resolve_type(Optional[str])
        assert schema == {"type": "string"}
        assert optional is True

    def test_optional_int(self):
        schema, optional = _resolve_type(Optional[int])
        assert schema == {"type": "integer"}
        assert optional is True

    def test_literal_strings(self):
        schema, optional = _resolve_type(Literal["a", "b", "c"])
        assert schema == {"type": "string", "enum": ["a", "b", "c"]}
        assert optional is False

    def test_literal_ints(self):
        schema, optional = _resolve_type(Literal[1, 2, 3])
        assert schema == {"type": "integer", "enum": [1, 2, 3]}
        assert optional is False

    def test_list_of_str(self):
        schema, optional = _resolve_type(List[str])
        assert schema == {"type": "array", "items": {"type": "string"}}
        assert optional is False

    def test_list_of_int(self):
        schema, optional = _resolve_type(List[int])
        assert schema == {"type": "array", "items": {"type": "integer"}}
        assert optional is False

    def test_dict_str_any(self):
        schema, optional = _resolve_type(Dict[str, int])
        assert schema == {"type": "object"}
        assert optional is False

    def test_unknown_type_falls_back_to_string(self):
        class CustomType:
            pass
        schema, optional = _resolve_type(CustomType)
        assert schema == {"type": "string"}
        assert optional is False


# ===========================================================================
# Tests for _parse_docstring_args
# ===========================================================================

class TestParseDocstringArgs:
    """Tests for docstring parsing."""

    def test_empty_docstring(self):
        summary, params = _parse_docstring_args("")
        assert summary == ""
        assert params == {}

    def test_none_docstring(self):
        summary, params = _parse_docstring_args(None)
        assert summary == ""
        assert params == {}

    def test_summary_only(self):
        summary, params = _parse_docstring_args("Get the weather forecast.")
        assert summary == "Get the weather forecast."
        assert params == {}

    def test_args_block(self):
        doc = """Get the weather forecast.

        Args:
            city: Name of the city
            units: Temperature units (celsius or fahrenheit)
        """
        summary, params = _parse_docstring_args(doc)
        assert summary == "Get the weather forecast."
        assert params["city"] == "Name of the city"
        assert params["units"] == "Temperature units (celsius or fahrenheit)"

    def test_args_block_with_type_annotations(self):
        doc = """Look up a user.

        Args:
            name (str): The user's name
            age (int): The user's age
        """
        summary, params = _parse_docstring_args(doc)
        assert summary == "Look up a user."
        assert params["name"] == "The user's name"
        assert params["age"] == "The user's age"

    def test_args_block_with_returns_section(self):
        doc = """Do something.

        Args:
            x: First param
            y: Second param

        Returns:
            Some result
        """
        summary, params = _parse_docstring_args(doc)
        assert params["x"] == "First param"
        assert params["y"] == "Second param"

    def test_multiline_param_description(self):
        doc = """Search function.

        Args:
            query: The search query to execute
                against the database
            limit: Maximum results
        """
        summary, params = _parse_docstring_args(doc)
        assert "search query to execute" in params["query"]
        assert "against the database" in params["query"]
        assert params["limit"] == "Maximum results"


# ===========================================================================
# Tests for infer_schema - detection heuristic
# ===========================================================================

class TestInferSchemaDetection:
    """Tests for when infer_schema should and should not activate."""

    def test_old_style_args_raw_data(self):
        """Old-style (args, raw_data) should not be treated as typed."""
        def handler(args, raw_data):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is False

    def test_old_style_args_only(self):
        """Old-style (args,) should not be treated as typed."""
        def handler(args):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is False

    def test_varargs_fallback(self):
        """Functions with *args should fall back to old style."""
        def handler(*args):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is False

    def test_kwargs_fallback(self):
        """Functions with **kwargs should fall back to old style."""
        def handler(**kwargs):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is False

    def test_typed_params_detected(self):
        """Typed parameters should be detected as new style."""
        def handler(city: str, units: str = "celsius"):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is True

    def test_zero_param_tool(self):
        """Function with no params (after self filtering) is a valid zero-param typed tool."""
        def handler():
            """Get the current time."""
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is True
        assert params == {}
        assert required == []
        assert desc == "Get the current time."

    def test_self_filtered_out(self):
        """self parameter should be filtered out."""
        def handler(self, city: str):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is True
        assert "self" not in params
        assert "city" in params

    def test_no_annotations_fallback(self):
        """No type hints on non-standard param names should fall back."""
        def handler(city, units):
            pass
        params, required, desc, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is False


# ===========================================================================
# Tests for infer_schema - type mapping
# ===========================================================================

class TestInferSchemaTypes:
    """Tests for correct type mapping in inferred schemas."""

    def test_string_param(self):
        def handler(name: str):
            pass
        params, *_ = infer_schema(handler)
        assert params["name"]["type"] == "string"

    def test_int_param(self):
        def handler(count: int):
            pass
        params, *_ = infer_schema(handler)
        assert params["count"]["type"] == "integer"

    def test_float_param(self):
        def handler(price: float):
            pass
        params, *_ = infer_schema(handler)
        assert params["price"]["type"] == "number"

    def test_bool_param(self):
        def handler(enabled: bool):
            pass
        params, *_ = infer_schema(handler)
        assert params["enabled"]["type"] == "boolean"

    def test_list_param(self):
        def handler(items: list):
            pass
        params, *_ = infer_schema(handler)
        assert params["items"]["type"] == "array"

    def test_dict_param(self):
        def handler(data: dict):
            pass
        params, *_ = infer_schema(handler)
        assert params["data"]["type"] == "object"

    def test_optional_param(self):
        def handler(name: str, nickname: Optional[str] = None):
            pass
        params, required, *_ = infer_schema(handler)
        assert params["name"]["type"] == "string"
        assert params["nickname"]["type"] == "string"
        assert "name" in required
        assert "nickname" not in required

    def test_literal_param(self):
        def handler(color: Literal["red", "green", "blue"]):
            pass
        params, *_ = infer_schema(handler)
        assert params["color"]["type"] == "string"
        assert params["color"]["enum"] == ["red", "green", "blue"]

    def test_list_of_str_param(self):
        def handler(tags: List[str]):
            pass
        params, *_ = infer_schema(handler)
        assert params["tags"]["type"] == "array"
        assert params["tags"]["items"] == {"type": "string"}


# ===========================================================================
# Tests for infer_schema - required/optional
# ===========================================================================

class TestInferSchemaRequired:
    """Tests for required vs optional parameter detection."""

    def test_no_default_is_required(self):
        def handler(city: str):
            pass
        _, required, *_ = infer_schema(handler)
        assert "city" in required

    def test_with_default_is_optional(self):
        def handler(city: str = "London"):
            pass
        _, required, *_ = infer_schema(handler)
        assert "city" not in required

    def test_optional_type_is_not_required(self):
        def handler(city: Optional[str]):
            pass
        _, required, *_ = infer_schema(handler)
        assert "city" not in required

    def test_mixed_required_optional(self):
        def handler(city: str, units: str = "celsius", country: Optional[str] = None):
            pass
        _, required, *_ = infer_schema(handler)
        assert "city" in required
        assert "units" not in required
        assert "country" not in required


# ===========================================================================
# Tests for infer_schema - docstring integration
# ===========================================================================

class TestInferSchemaDocstring:
    """Tests for docstring-driven description and parameter docs."""

    def test_description_from_docstring(self):
        def handler(city: str):
            """Get the weather forecast."""
            pass
        _, _, desc, *_ = infer_schema(handler)
        assert desc == "Get the weather forecast."

    def test_no_docstring(self):
        def handler(city: str):
            pass
        _, _, desc, *_ = infer_schema(handler)
        assert desc is None

    def test_param_descriptions_from_docstring(self):
        def handler(city: str, units: str = "celsius"):
            """Get the weather.

            Args:
                city: Name of the city
                units: Temperature units
            """
            pass
        params, *_ = infer_schema(handler)
        assert params["city"]["description"] == "Name of the city"
        assert params["units"]["description"] == "Temperature units"


# ===========================================================================
# Tests for infer_schema - raw_data handling
# ===========================================================================

class TestInferSchemaRawData:
    """Tests for raw_data parameter detection and exclusion."""

    def test_raw_data_detected(self):
        def handler(city: str, raw_data: dict = None):
            pass
        _, _, _, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is True
        assert has_raw_data is True

    def test_raw_data_excluded_from_schema(self):
        def handler(city: str, raw_data: dict = None):
            pass
        params, *_ = infer_schema(handler)
        assert "raw_data" not in params
        assert "city" in params

    def test_no_raw_data(self):
        def handler(city: str):
            pass
        _, _, _, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is True
        assert has_raw_data is False

    def test_only_raw_data(self):
        """Function with only raw_data param is a zero-param typed tool with raw_data."""
        def handler(raw_data: dict):
            pass
        params, required, _, is_typed, has_raw_data = infer_schema(handler)
        assert is_typed is True
        assert has_raw_data is True
        assert params == {}
        assert required == []


# ===========================================================================
# Tests for create_typed_handler_wrapper
# ===========================================================================

class TestCreateTypedHandlerWrapper:
    """Tests for the handler wrapper function."""

    def test_wrapper_unpacks_args(self):
        def handler(city: str, units: str = "celsius"):
            return SwaigFunctionResult(f"Weather in {city} ({units})")

        wrapper = create_typed_handler_wrapper(handler, has_raw_data=False)
        result = wrapper({"city": "London", "units": "fahrenheit"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "London" in result.response
        assert "fahrenheit" in result.response

    def test_wrapper_passes_raw_data(self):
        def handler(city: str, raw_data: dict = None):
            return SwaigFunctionResult(f"Weather in {city}, call={raw_data.get('call_id', 'none')}")

        wrapper = create_typed_handler_wrapper(handler, has_raw_data=True)
        result = wrapper({"city": "Paris"}, {"call_id": "abc123"})
        assert isinstance(result, SwaigFunctionResult)
        assert "Paris" in result.response
        assert "abc123" in result.response

    def test_wrapper_without_raw_data(self):
        def handler(name: str):
            return SwaigFunctionResult(f"Hello, {name}")

        wrapper = create_typed_handler_wrapper(handler, has_raw_data=False)
        result = wrapper({"name": "Alice"}, {"call_id": "ignored"})
        assert isinstance(result, SwaigFunctionResult)
        assert "Alice" in result.response

    def test_wrapper_preserves_name(self):
        def my_tool(x: str):
            pass

        wrapper = create_typed_handler_wrapper(my_tool, has_raw_data=False)
        assert wrapper.__name__ == "my_tool"

    def test_wrapper_preserves_doc(self):
        def my_tool(x: str):
            """My tool docstring."""
            pass

        wrapper = create_typed_handler_wrapper(my_tool, has_raw_data=False)
        assert wrapper.__doc__ == "My tool docstring."

    def test_wrapper_has_wrapped_attr(self):
        def my_tool(x: str):
            pass

        wrapper = create_typed_handler_wrapper(my_tool, has_raw_data=False)
        assert wrapper.__wrapped__ is my_tool

    def test_wrapper_with_empty_args(self):
        def handler():
            return SwaigFunctionResult("done")

        wrapper = create_typed_handler_wrapper(handler, has_raw_data=False)
        result = wrapper({}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert result.response == "done"

    def test_wrapper_with_default_values(self):
        def handler(city: str, units: str = "celsius"):
            return SwaigFunctionResult(f"{city}:{units}")

        wrapper = create_typed_handler_wrapper(handler, has_raw_data=False)
        # Only pass required arg, default should fill in
        result = wrapper({"city": "Tokyo"}, {})
        assert "Tokyo" in result.response
        assert "celsius" in result.response


# ===========================================================================
# Tests for end-to-end integration with AgentBase
# ===========================================================================

class TestEndToEndIntegration:
    """Tests that type inference works end-to-end through the decorator and registry."""

    def test_class_decorator_with_typed_handler(self):
        """Test that @AgentBase.tool() with typed params infers schema correctly."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test")

            @AgentBase.tool(name="get_weather")
            def get_weather(self, city: str, units: str = "celsius"):
                """Get the weather forecast.

                Args:
                    city: Name of the city
                    units: Temperature units
                """
                return SwaigFunctionResult(f"Weather in {city}")

        agent = TestAgent()
        func = agent._tool_registry.get_function("get_weather")
        assert func is not None
        assert func.is_typed_handler is True
        assert "city" in func.parameters
        assert func.parameters["city"]["type"] == "string"
        assert func.parameters["city"]["description"] == "Name of the city"
        assert "units" in func.parameters
        assert func.parameters["units"]["type"] == "string"
        assert "city" in func.required
        assert "units" not in func.required
        assert func.description == "Get the weather forecast."

    def test_class_decorator_with_explicit_params_not_overridden(self):
        """Explicit parameters= should always win over inference."""
        from signalwire_agents import AgentBase

        explicit_params = {
            "location": {"type": "string", "description": "Location name"}
        }

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test2")

            @AgentBase.tool(name="get_weather2", parameters=explicit_params, description="Explicit desc")
            def get_weather(self, city: str, units: str = "celsius"):
                """Get the weather forecast."""
                return SwaigFunctionResult(f"Weather in {city}")

        agent = TestAgent()
        func = agent._tool_registry.get_function("get_weather2")
        assert func is not None
        assert func.is_typed_handler is False
        assert "location" in func.parameters
        assert "city" not in func.parameters
        assert func.description == "Explicit desc"

    def test_class_decorator_old_style_still_works(self):
        """Old-style (self, args, raw_data) should work as before."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test3")

            @AgentBase.tool(
                name="old_tool",
                description="Old style tool",
                parameters={"x": {"type": "string", "description": "Input"}},
            )
            def old_tool(self, args, raw_data):
                return SwaigFunctionResult("done")

        agent = TestAgent()
        func = agent._tool_registry.get_function("old_tool")
        assert func is not None
        assert func.is_typed_handler is False
        assert func.parameters == {"x": {"type": "string", "description": "Input"}}

    def test_typed_handler_execution_through_on_function_call(self):
        """Typed handler should execute correctly through the standard dispatch."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test4")

            @AgentBase.tool(name="greet")
            def greet(self, name: str, greeting: str = "Hello"):
                """Greet someone.

                Args:
                    name: Person's name
                    greeting: Greeting word
                """
                return SwaigFunctionResult(f"{greeting}, {name}!")

        agent = TestAgent()
        result = agent.on_function_call("greet", {"name": "Alice", "greeting": "Hi"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "Hi" in result.response
        assert "Alice" in result.response

    def test_typed_handler_with_defaults(self):
        """Typed handler should use defaults for missing args."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test5")

            @AgentBase.tool(name="greet2")
            def greet(self, name: str, greeting: str = "Hello"):
                """Greet someone."""
                return SwaigFunctionResult(f"{greeting}, {name}!")

        agent = TestAgent()
        result = agent.on_function_call("greet2", {"name": "Bob"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "Hello" in result.response
        assert "Bob" in result.response

    def test_typed_handler_with_raw_data(self):
        """Typed handler with raw_data should receive it."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test6")

            @AgentBase.tool(name="check_call")
            def check_call(self, query: str, raw_data: dict = None):
                """Check the call.

                Args:
                    query: Search query
                """
                call_id = raw_data.get("call_id", "unknown") if raw_data else "unknown"
                return SwaigFunctionResult(f"query={query}, call={call_id}")

        agent = TestAgent()
        result = agent.on_function_call("check_call", {"query": "test"}, {"call_id": "c42"})
        assert isinstance(result, SwaigFunctionResult)
        assert "test" in result.response
        assert "c42" in result.response

    def test_literal_enum_in_swml(self):
        """Literal types should produce enum in SWML output."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test7")

            @AgentBase.tool(name="set_mode")
            def set_mode(self, mode: Literal["auto", "manual", "off"]):
                """Set the operating mode.

                Args:
                    mode: The mode to set
                """
                return SwaigFunctionResult(f"Mode set to {mode}")

        agent = TestAgent()
        func = agent._tool_registry.get_function("set_mode")
        assert func.parameters["mode"]["type"] == "string"
        assert func.parameters["mode"]["enum"] == ["auto", "manual", "off"]

    def test_instance_decorator_with_typed_handler(self):
        """Test the instance-level @agent.tool() decorator with typed params."""
        from signalwire_agents import AgentBase

        agent = AgentBase("Test Agent", route="/test8")

        @agent._tool_decorator(name="lookup")
        def lookup(user_id: int, include_details: bool = False):
            """Look up a user.

            Args:
                user_id: The user's ID
                include_details: Whether to include extra details
            """
            return SwaigFunctionResult(f"User {user_id}")

        func = agent._tool_registry.get_function("lookup")
        assert func is not None
        assert func.is_typed_handler is True
        assert func.parameters["user_id"]["type"] == "integer"
        assert func.parameters["include_details"]["type"] == "boolean"
        assert "user_id" in func.required
        assert "include_details" not in func.required
        assert func.description == "Look up a user."

    def test_swml_output_includes_inferred_schema(self):
        """The generated SWML should include the inferred parameter schema."""
        from signalwire_agents import AgentBase

        class TestAgent(AgentBase):
            def __init__(self):
                super().__init__("Test Agent", route="/test9")

            @AgentBase.tool(name="search")
            def search(self, query: str, limit: int = 10):
                """Search for items.

                Args:
                    query: Search query string
                    limit: Max results to return
                """
                return SwaigFunctionResult("results")

        agent = TestAgent()
        func = agent._tool_registry.get_function("search")
        swaig = func.to_swaig("https://example.com")

        assert swaig["function"] == "search"
        assert swaig["description"] == "Search for items."
        params = swaig["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "limit" in params["properties"]
        assert params["properties"]["limit"]["type"] == "integer"
        assert "query" in params.get("required", [])
        assert "limit" not in params.get("required", [])
