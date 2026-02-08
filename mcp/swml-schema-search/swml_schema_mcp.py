#!/usr/bin/env python3
"""
SWML Schema MCP Server - Query SWML schema definitions

Provides tools to list, search, and retrieve SWML method definitions
from the schema file without loading the entire 385KB file.
"""

import sys
import json
import os
import logging
from typing import Dict, Any, Optional, List

# Configure logging
if os.environ.get('SWML_SCHEMA_MCP_DEBUG'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
else:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
logger = logging.getLogger(__name__)

# Schema data loaded at startup
SCHEMA: Dict[str, Any] = {}
METHODS: Dict[str, Dict[str, Any]] = {}  # method_name -> definition
METHOD_DESCRIPTIONS: Dict[str, str] = {}  # method_name -> brief description


def load_schema():
    """Load and index the SWML schema file."""
    global SCHEMA, METHODS, METHOD_DESCRIPTIONS

    # Default to the schema.json in the signalwire_agents package
    default_schema_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'signalwire_agents', 'schema.json'
    )
    schema_path = os.environ.get('SWML_SCHEMA_PATH', default_schema_path)
    schema_path = os.path.normpath(schema_path)

    logger.info(f"Loading schema from {schema_path}")

    with open(schema_path, 'r') as f:
        SCHEMA = json.load(f)

    defs = SCHEMA.get('$defs', {})

    # Find actual SWML methods from SWMLMethod.anyOf
    swml_method_def = defs.get('SWMLMethod', {})
    method_refs = swml_method_def.get('anyOf', [])

    for ref in method_refs:
        ref_path = ref.get('$ref', '')
        # Extract definition name from "#/$defs/MethodName"
        if ref_path.startswith('#/$defs/'):
            def_name = ref_path.split('/')[-1]
            if def_name in defs:
                definition = defs[def_name]
                # Get the actual method name (lowercase key inside properties)
                props = definition.get('properties', {})
                if props:
                    method_name = list(props.keys())[0]
                    METHODS[method_name] = definition

                    # Extract description
                    method_props = props.get(method_name, {})
                    desc = method_props.get('description', '')
                    # Take first line/sentence
                    if desc:
                        desc = desc.split('\n')[0].strip()
                        if len(desc) > 100:
                            desc = desc[:97] + '...'
                    METHOD_DESCRIPTIONS[method_name] = desc or f"SWML {method_name} method"

    logger.info(f"Loaded {len(METHODS)} SWML methods")


def get_method_details(method_name: str) -> Optional[Dict[str, Any]]:
    """Get details for a method. Returns raw schema without deep ref resolution."""
    if method_name not in METHODS:
        return None

    definition = METHODS[method_name]
    defs = SCHEMA.get('$defs', {})

    # Only resolve the immediate top-level refs, not recursively
    def resolve_shallow(obj):
        if isinstance(obj, dict):
            if '$ref' in obj and len(obj) == 1:
                ref_path = obj['$ref']
                if ref_path.startswith('#/$defs/'):
                    ref_name = ref_path.split('/')[-1]
                    if ref_name in defs:
                        # Return ref name as hint, don't expand
                        return {"_ref": ref_name, "_hint": "Use get_swml_method with this ref name for details"}
            # Only go one level deep for the method's own properties
            return {k: v for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_shallow(item) for item in obj]
        return obj

    # Get the method's properties with shallow ref hints
    result = {}
    for key, value in definition.items():
        if key == 'properties':
            # Expand the method's direct properties
            result[key] = {}
            for prop_name, prop_value in value.items():
                if isinstance(prop_value, dict) and 'properties' in prop_value:
                    # This is the actual method object, show its properties
                    result[key][prop_name] = {
                        'description': prop_value.get('description', ''),
                        'properties': {}
                    }
                    for inner_name, inner_value in prop_value.get('properties', {}).items():
                        if isinstance(inner_value, dict):
                            inner_copy = dict(inner_value)
                            # Replace $refs with readable hints
                            if '$ref' in inner_copy:
                                ref_name = inner_copy['$ref'].split('/')[-1]
                                inner_copy['_type'] = ref_name
                                del inner_copy['$ref']
                            if 'anyOf' in inner_copy:
                                types = []
                                for item in inner_copy['anyOf']:
                                    if '$ref' in item:
                                        types.append(item['$ref'].split('/')[-1])
                                    elif 'type' in item:
                                        types.append(item['type'])
                                inner_copy['_types'] = types
                                del inner_copy['anyOf']
                            if 'oneOf' in inner_copy:
                                types = []
                                for item in inner_copy['oneOf']:
                                    if '$ref' in item:
                                        types.append(item['$ref'].split('/')[-1])
                                    elif 'type' in item:
                                        types.append(item['type'])
                                inner_copy['_oneOf'] = types
                                del inner_copy['oneOf']
                            result[key][prop_name]['properties'][inner_name] = inner_copy
                        else:
                            result[key][prop_name]['properties'][inner_name] = inner_value
                elif isinstance(prop_value, dict) and 'oneOf' in prop_value:
                    # Method has variants (like connect)
                    variants = []
                    for item in prop_value['oneOf']:
                        if '$ref' in item:
                            variants.append(item['$ref'].split('/')[-1])
                    result[key][prop_name] = {
                        'description': prop_value.get('description', ''),
                        '_variants': variants,
                        '_hint': 'Multiple forms available - see variant names'
                    }
                elif isinstance(prop_value, dict) and '$ref' in prop_value:
                    # Method references another definition (like ai -> AIObject)
                    ref_name = prop_value['$ref'].split('/')[-1]
                    ref_def = defs.get(ref_name, {})
                    # Get the properties from the referenced definition
                    ref_props = ref_def.get('properties', {})
                    prop_summary = {}
                    for rp_name, rp_value in ref_props.items():
                        if isinstance(rp_value, dict):
                            prop_info = {'description': rp_value.get('description', '')}
                            if 'type' in rp_value:
                                prop_info['type'] = rp_value['type']
                            if '$ref' in rp_value:
                                prop_info['_type'] = rp_value['$ref'].split('/')[-1]
                            if 'anyOf' in rp_value:
                                prop_info['_types'] = [
                                    i.get('$ref', '').split('/')[-1] or i.get('type', '')
                                    for i in rp_value['anyOf']
                                ]
                            if 'oneOf' in rp_value:
                                prop_info['_oneOf'] = [
                                    i.get('$ref', '').split('/')[-1] or i.get('type', '')
                                    for i in rp_value['oneOf']
                                ]
                            prop_summary[rp_name] = prop_info
                    result[key][prop_name] = {
                        'description': prop_value.get('description', ''),
                        '_definedIn': ref_name,
                        'properties': prop_summary
                    }
                else:
                    result[key][prop_name] = prop_value
        else:
            result[key] = value

    return result


def search_methods(keyword: str) -> List[Dict[str, str]]:
    """Search methods by keyword in name or description."""
    keyword_lower = keyword.lower()
    results = []

    for method_name, description in METHOD_DESCRIPTIONS.items():
        if keyword_lower in method_name.lower() or keyword_lower in description.lower():
            results.append({
                'name': method_name,
                'description': description
            })

    return sorted(results, key=lambda x: x['name'])


def send_message(message: Dict[str, Any]) -> None:
    """Send a JSON-RPC message via stdout."""
    sys.stdout.write(json.dumps(message) + '\n')
    sys.stdout.flush()


def read_message() -> Optional[Dict[str, Any]]:
    """Read a JSON-RPC message from stdin."""
    try:
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def main():
    # Load schema at startup
    try:
        load_schema()
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        sys.exit(1)

    while True:
        message = read_message()
        if message is None:
            break

        method = message.get("method")
        request_id = message.get("id")

        try:
            if method == "initialize":
                logger.info("SWML Schema MCP server initialized")

                send_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "swml-schema-mcp",
                            "version": "1.0.0"
                        }
                    }
                })

            elif method == "tools/list":
                send_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "list_swml_methods",
                                "description": "List all available SWML methods with brief descriptions.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            },
                            {
                                "name": "get_swml_method",
                                "description": "Get the full schema definition for a specific SWML method including all parameters, types, and descriptions.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "method_name": {
                                            "type": "string",
                                            "description": "The SWML method name (e.g., 'ai', 'connect', 'play')"
                                        }
                                    },
                                    "required": ["method_name"]
                                }
                            },
                            {
                                "name": "search_swml_methods",
                                "description": "Search SWML methods by keyword in name or description.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "keyword": {
                                            "type": "string",
                                            "description": "Keyword to search for"
                                        }
                                    },
                                    "required": ["keyword"]
                                }
                            }
                        ]
                    }
                })

            elif method == "tools/call":
                params = message.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name == "list_swml_methods":
                    output_lines = [f"Available SWML Methods ({len(METHODS)} total):\n"]

                    for method_name in sorted(METHODS.keys()):
                        desc = METHOD_DESCRIPTIONS.get(method_name, '')
                        output_lines.append(f"  {method_name}")
                        if desc:
                            output_lines.append(f"    {desc}")
                        output_lines.append("")

                    send_message({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "\n".join(output_lines)
                                }
                            ]
                        }
                    })

                elif tool_name == "get_swml_method":
                    method_name = arguments.get("method_name", "").lower()

                    if not method_name:
                        raise ValueError("method_name parameter is required")

                    details = get_method_details(method_name)

                    if details is None:
                        # Try to find close matches
                        matches = [m for m in METHODS.keys() if method_name in m.lower()]
                        if matches:
                            raise ValueError(f"Method '{method_name}' not found. Did you mean: {', '.join(matches)}?")
                        else:
                            raise ValueError(f"Method '{method_name}' not found. Use list_swml_methods to see available methods.")

                    output = f"SWML Method: {method_name}\n"
                    output += f"Description: {METHOD_DESCRIPTIONS.get(method_name, 'N/A')}\n\n"
                    output += "Schema Definition:\n"
                    output += json.dumps(details, indent=2)

                    send_message({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": output
                                }
                            ]
                        }
                    })

                elif tool_name == "search_swml_methods":
                    keyword = arguments.get("keyword", "")

                    if not keyword:
                        raise ValueError("keyword parameter is required")

                    results = search_methods(keyword)

                    if not results:
                        output = f"No methods found matching '{keyword}'."
                    else:
                        output_lines = [f"Methods matching '{keyword}' ({len(results)} found):\n"]
                        for r in results:
                            output_lines.append(f"  {r['name']}")
                            if r['description']:
                                output_lines.append(f"    {r['description']}")
                            output_lines.append("")
                        output = "\n".join(output_lines)

                    send_message({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": output
                                }
                            ]
                        }
                    })

                else:
                    send_message({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    })

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            send_message({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            })


if __name__ == "__main__":
    main()
