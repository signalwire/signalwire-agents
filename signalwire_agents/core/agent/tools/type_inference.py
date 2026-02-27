"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""Type-hint-based schema inference for SWAIG tool functions."""

import inspect
import re
import typing
from typing import Any, Dict, List, Optional, Tuple, get_type_hints


# Map Python types to JSON Schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _resolve_type(annotation) -> Tuple[Dict[str, Any], bool]:
    """
    Resolve a Python type annotation to a JSON Schema property dict.

    Returns:
        (schema_dict, is_optional) where is_optional indicates if the
        parameter is Optional[X] (i.e. Union[X, None]).
    """
    origin = getattr(annotation, "__origin__", None)

    # Handle Optional[X] which is Union[X, None]
    if origin is typing.Union:
        args = annotation.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            # This is Optional[X]
            inner_schema, _ = _resolve_type(non_none[0])
            return inner_schema, True
        # General Union — fall back to string
        return {"type": "string"}, False

    # Handle Literal["a", "b", ...]
    if origin is typing.Literal:
        values = list(annotation.__args__)
        # Determine type from first value
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}, False
        elif all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}, False
        else:
            return {"type": "string", "enum": [str(v) for v in values]}, False

    # Handle List[X]
    if origin is list:
        args = getattr(annotation, "__args__", None)
        if args:
            items_schema, _ = _resolve_type(args[0])
            return {"type": "array", "items": items_schema}, False
        return {"type": "array"}, False

    # Handle Dict[K, V]
    if origin is dict:
        return {"type": "object"}, False

    # Direct type lookup
    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}, False

    # Unknown type — fall back to string
    return {"type": "string"}, False


def _parse_docstring_args(docstring: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a Google-style docstring to extract the summary line
    and per-parameter descriptions from the Args: block.

    Returns:
        (summary, param_descriptions) where summary is the first non-empty
        line and param_descriptions maps parameter names to their descriptions.
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().splitlines()

    # Summary is the first non-blank line
    summary = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            summary = stripped
            break

    # Find the Args: block
    param_descriptions = {}
    in_args = False
    current_param = None
    current_desc_lines = []

    for line in lines:
        stripped = line.strip()

        # Detect start of Args block
        if stripped.lower().startswith("args:"):
            in_args = True
            continue

        # Detect end of Args block (another section header like Returns:, Raises:, etc.)
        if in_args and stripped and stripped.endswith(":") and not stripped.startswith("-"):
            # Check if this looks like a section header (single word followed by colon)
            maybe_section = stripped.rstrip(":")
            if " " not in maybe_section and maybe_section[0].isupper():
                # Flush current param
                if current_param:
                    param_descriptions[current_param] = " ".join(current_desc_lines).strip()
                break

        if in_args:
            # Match parameter line: "  param_name: description" or "  param_name (type): description"
            match = re.match(r'^\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)', line)
            if match:
                # Flush previous param
                if current_param:
                    param_descriptions[current_param] = " ".join(current_desc_lines).strip()
                current_param = match.group(1)
                current_desc_lines = [match.group(2).strip()] if match.group(2).strip() else []
            elif current_param and stripped:
                # Continuation line for current param
                current_desc_lines.append(stripped)

    # Flush last param
    if current_param:
        param_descriptions[current_param] = " ".join(current_desc_lines).strip()

    return summary, param_descriptions


def infer_schema(func) -> Tuple[Dict[str, Dict], List[str], Optional[str], bool, bool]:
    """
    Inspect a function's signature and type hints to infer a JSON Schema
    for SWAIG tool parameters.

    Args:
        func: The function to inspect.

    Returns:
        A tuple of (parameters, required, description, is_typed, has_raw_data):
        - parameters: dict mapping parameter names to JSON Schema property dicts
        - required: list of required parameter names
        - description: tool description from docstring (or None)
        - is_typed: True if the function uses typed parameters (new style)
        - has_raw_data: True if the function accepts a `raw_data` parameter
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Filter out 'self' (bound method or unbound in class body)
    params = [p for p in params if p.name != "self"]

    # Check for *args or **kwargs — fall back to old style
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return {}, [], None, False, False

    # Check if this is the old-style (args, raw_data) pattern
    # Old-style: exactly (args, raw_data) or just (args,) with no type hints
    param_names = [p.name for p in params]
    if param_names == ["args", "raw_data"] or param_names == ["args"]:
        # Check if any have type annotations that aren't basic dict/Any
        has_meaningful_hints = False
        for p in params:
            if p.annotation is not inspect.Parameter.empty and p.annotation not in (dict, Dict, Any, Dict[str, Any]):
                has_meaningful_hints = True
                break
        if not has_meaningful_hints:
            return {}, [], None, False, False

    # If no params at all, this is a valid zero-param typed tool
    if not params:
        docstring = inspect.getdoc(func)
        summary, _ = _parse_docstring_args(docstring) if docstring else ("", {})
        return {}, [], summary or None, True, False

    # Check for raw_data parameter
    has_raw_data = any(p.name == "raw_data" for p in params)

    # Filter out raw_data from the schema params
    schema_params = [p for p in params if p.name != "raw_data"]

    # If no schema params remain after filtering raw_data, it's a zero-param typed tool
    if not schema_params:
        docstring = inspect.getdoc(func)
        summary, _ = _parse_docstring_args(docstring) if docstring else ("", {})
        return {}, [], summary or None, True, has_raw_data

    # Check if parameters have type annotations
    has_annotations = any(p.annotation is not inspect.Parameter.empty for p in schema_params)
    if not has_annotations:
        # No type hints — can't infer schema, fall back
        return {}, [], None, False, False

    # Try to get type hints (resolves forward references)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {p.name: p.annotation for p in schema_params if p.annotation is not inspect.Parameter.empty}

    # Parse docstring for description and per-param descriptions
    docstring = inspect.getdoc(func)
    summary, param_docs = _parse_docstring_args(docstring) if docstring else ("", {})

    # Build the schema
    parameters = {}
    required = []

    for p in schema_params:
        annotation = hints.get(p.name, inspect.Parameter.empty)

        if annotation is inspect.Parameter.empty:
            # No type hint for this param — default to string
            prop = {"type": "string"}
            is_optional = False
        else:
            prop, is_optional = _resolve_type(annotation)

        # Add description from docstring if available
        if p.name in param_docs and param_docs[p.name]:
            prop["description"] = param_docs[p.name]

        parameters[p.name] = prop

        # Determine if required: no default and not Optional → required
        if p.default is inspect.Parameter.empty and not is_optional:
            required.append(p.name)

    return parameters, required, summary or None, True, has_raw_data


def create_typed_handler_wrapper(func, has_raw_data: bool):
    """
    Wrap a typed handler function so it can be called with the standard
    SWAIG calling convention (args_dict, raw_data).

    The wrapper unpacks the args dict into keyword arguments for the
    original function.

    Args:
        func: The original typed handler function.
        has_raw_data: If True, pass raw_data as a keyword argument.

    Returns:
        A wrapper function with signature (args, raw_data).
    """
    def wrapper(args, raw_data):
        if has_raw_data:
            return func(raw_data=raw_data, **args)
        else:
            return func(**args)

    # Preserve original function metadata for debugging
    wrapper.__name__ = getattr(func, "__name__", "typed_handler")
    wrapper.__doc__ = getattr(func, "__doc__", None)
    wrapper.__wrapped__ = func

    return wrapper
