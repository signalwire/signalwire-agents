# SWML Schema MCP Server

An MCP (Model Context Protocol) server that provides tools to query SWML schema definitions. This allows LLMs to efficiently look up SWML method specifications without loading the entire schema file into context.

## Features

- **List all SWML methods** with brief descriptions
- **Get detailed schema** for any specific method
- **Search methods** by keyword in name or description
- Efficient on-demand access to the 385KB+ schema file

## Tools Provided

| Tool | Description |
|------|-------------|
| `list_swml_methods` | List all available SWML methods with brief descriptions |
| `get_swml_method` | Get the full schema definition for a specific method (e.g., 'ai', 'connect', 'play') |
| `search_swml_methods` | Search SWML methods by keyword in name or description |

## Installation

No additional dependencies required beyond Python 3.7+. The server uses only standard library modules.

## Configuration

### Schema Path

By default, the server uses the in-tree schema from the signalwire_agents package:

```
signalwire_agents/schema.json
```

To use a different schema file, set the `SWML_SCHEMA_PATH` environment variable:

```bash
export SWML_SCHEMA_PATH=/path/to/your/swml-schema.json
```

### Debug Logging

Enable debug logging by setting:

```bash
export SWML_SCHEMA_MCP_DEBUG=1
```

## Usage with Claude Code

Add to your Claude Code MCP settings (`~/.claude/claude_desktop_config.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "swml-schema": {
      "command": "python",
      "args": ["/path/to/signalwire-agents/mcp/swml-schema-search/swml_schema_mcp.py"]
    }
  }
}
```

Or with uv:

```json
{
  "mcpServers": {
    "swml-schema": {
      "command": "uv",
      "args": ["run", "python", "/path/to/signalwire-agents/mcp/swml-schema-search/swml_schema_mcp.py"]
    }
  }
}
```

## Example Tool Usage

### List all methods

```
Tool: list_swml_methods
Arguments: {}

Output:
Available SWML Methods (25 total):

  ai
    Starts an AI agent session with configurable prompts, functions, and behaviors

  connect
    Connects the call to another destination

  play
    Plays audio files or text-to-speech
  ...
```

### Get method details

```
Tool: get_swml_method
Arguments: {"method_name": "ai"}

Output:
SWML Method: ai
Description: Starts an AI agent session with configurable prompts, functions, and behaviors

Schema Definition:
{
  "properties": {
    "ai": {
      "description": "...",
      "properties": {
        "prompt": {...},
        "post_prompt": {...},
        "SWAIG": {...},
        ...
      }
    }
  }
}
```

### Search methods

```
Tool: search_swml_methods
Arguments: {"keyword": "audio"}

Output:
Methods matching 'audio' (3 found):

  play
    Plays audio files or text-to-speech

  record
    Records audio from the call

  play_background
    Plays background audio during the call
```

## How It Works

1. **Startup**: Loads and indexes the SWML schema JSON file
2. **Indexing**: Extracts all method definitions from `$defs.SWMLMethod.anyOf`
3. **Protocol**: Communicates via JSON-RPC over stdin/stdout (MCP standard)
4. **Efficiency**: Returns method details with shallow reference resolution, showing `_type` hints instead of fully expanding nested definitions

## Development

Run the server directly for testing:

```bash
python swml_schema_mcp.py
```

Then send JSON-RPC messages via stdin:

```json
{"jsonrpc": "2.0", "method": "initialize", "id": 1}
{"jsonrpc": "2.0", "method": "tools/list", "id": 2}
{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "list_swml_methods", "arguments": {}}, "id": 3}
```
