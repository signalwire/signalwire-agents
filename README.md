<!-- Header -->
<div align="center">
    <a href="https://signalwire.com" target="_blank">
        <img src="https://github.com/user-attachments/assets/0c8ed3b9-8c50-4dc6-9cc4-cc6cd137fd50" width="500" />
    </a>

# Agents SDK

_A Python SDK for creating, hosting, and securing SignalWire AI agents as microservices._

<p align="center">
  <a href="https://developer.signalwire.com/sdks/agents-sdk" target="_blank">Documentation</a> &middot;
  <a href="https://github.com/signalwire/signalwire-docs/issues/new/choose" target="_blank">Report an Issue</a> &middot;
  <a href="https://pypi.org/project/signalwire-agents/" target="_blank">PyPI</a>
</p>

<a href="https://discord.com/invite/F2WNYTNjuF" target="_blank"><img src="https://img.shields.io/badge/Discord%20Community-5865F2" alt="Discord" /></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/MIT-License-blue" alt="MIT License" /></a>
<a href="https://github.com/signalwire/signalwire-agents" target="_blank"><img src="https://img.shields.io/github/stars/signalwire/signalwire-agents" alt="GitHub Stars" /></a>

</div>

---

## Quick Start

```bash
pip install signalwire-agents
```

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent", route="/agent")

        self.add_language(name="English", code="en-US", voice="inworld.Mark")
        self.prompt_add_section("Role", body="You are a helpful assistant.")

    @AgentBase.tool("Get the current time")
    def get_time(self):
        """Get the current time"""
        from datetime import datetime
        return SwaigFunctionResult(f"The time is {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    agent = MyAgent()
    agent.run()
```

Test locally without running a server:

```bash
swaig-test my_agent.py --list-tools
swaig-test my_agent.py --dump-swml
swaig-test my_agent.py --exec get_time
```

## Features

- **Self-contained agents** -- each agent is both a web application and an AI persona
- **Prompt Object Model (POM)** -- structured prompt composition via `prompt_add_section()`
- **SWAIG tools** -- [SWAIG (SignalWire AI Gateway)](docs/swaig_reference.md) is the platform's AI tool-calling system with native access to the media stack; define functions with `@AgentBase.tool()` decorators and the AI can invoke them mid-call
- **Skills system** -- add capabilities with one-liners: `agent.add_skill("datetime")`
- **Contexts and steps** -- structured multi-step workflows with navigation control
- **DataMap tools** -- define tools that execute on SignalWire's servers, calling REST APIs without needing your own webhook endpoints
- **Dynamic configuration** -- per-request agent customization for multi-tenant deployments
- **Call flow control** -- pre-answer, post-answer, and post-AI verb insertion
- **Prefab agents** -- ready-to-use archetypes (InfoGatherer, Survey, FAQ, Receptionist, Concierge)
- **Multi-agent hosting** -- serve multiple agents on a single server with `AgentServer`
- **Local search** -- offline document search with vector similarity and keyword matching
- **SIP routing** -- route SIP calls to agents based on usernames
- **Session state** -- persistent conversation state with global data and post-prompt summaries
- **Security** -- auto-generated basic auth, function-specific tokens, SSL support
- **Serverless deployment** -- auto-detects Lambda, CGI, Google Cloud Functions, Azure Functions

## Installation

```bash
# Core SDK
pip install signalwire-agents

# With search (pick one based on your needs)
pip install signalwire-agents[search-queryonly]   # Query pre-built .swsearch files (~400MB)
pip install signalwire-agents[search]              # Build + query search indexes (~500MB)
pip install signalwire-agents[search-full]         # + PDF, DOCX, Excel, HTML processing (~600MB)
pip install signalwire-agents[search-all]          # All search features (~700MB)
```

## Examples

The [`examples/`](examples/) directory contains 50+ working examples. A few starting points:

| Example | What it demonstrates |
|---------|---------------------|
| [simple_agent.py](examples/simple_agent.py) | POM prompts, SWAIG tools, multilingual support, LLM tuning |
| [contexts_demo.py](examples/contexts_demo.py) | Multi-persona workflow with context switching and step navigation |
| [data_map_demo.py](examples/data_map_demo.py) | Server-side API tools without webhooks |
| [skills_demo.py](examples/skills_demo.py) | Loading built-in skills (datetime, math) |
| [call_flow_and_actions_demo.py](examples/call_flow_and_actions_demo.py) | Call flow verbs, debug events, SwaigFunctionResult actions |
| [session_and_state_demo.py](examples/session_and_state_demo.py) | on_summary, global data, post-prompt summaries |
| [swaig_features_agent.py](examples/swaig_features_agent.py) | Type inference, fillers, default webhook URLs |
| [multi_agent_server.py](examples/multi_agent_server.py) | Multiple agents on one server |
| [lambda_agent.py](examples/lambda_agent.py) | AWS Lambda deployment with Mangum |
| [comprehensive_dynamic_agent.py](examples/comprehensive_dynamic_agent.py) | Per-request dynamic configuration, multi-tenant routing |

See [examples/README.md](examples/README.md) for the full list organized by category.

Run any example:

```bash
python examples/simple_agent.py

# Or test without running a server
swaig-test examples/simple_agent.py --list-tools
swaig-test examples/simple_agent.py --dump-swml
swaig-test examples/simple_agent.py --exec get_weather --location "New York"
```

## Documentation

Full reference documentation is available at **[developer.signalwire.com/sdks/agents-sdk](https://developer.signalwire.com/sdks/agents-sdk)**.

Guides are also available in the [`docs/`](docs/) directory:

### Getting Started

- [Agent Guide](docs/agent_guide.md) -- creating agents, prompt configuration, dynamic setup
- [Architecture](docs/architecture.md) -- SDK architecture and core concepts
- [SDK Features](docs/sdk_features.md) -- feature overview, SDK vs raw SWML (SignalWire Markup Language) comparison

### Core Features

- [SWAIG Reference](docs/swaig_reference.md) -- function results, actions, post_data lifecycle
- [Contexts and Steps](docs/contexts_guide.md) -- structured workflows, navigation, gather mode
- [DataMap Guide](docs/datamap_guide.md) -- serverless API tools without webhooks
- [LLM Parameters](docs/llm_parameters.md) -- temperature, top_p, barge confidence tuning
- [SWML Service Guide](docs/swml_service_guide.md) -- low-level construction of SWML documents (the JSON format that defines agent behavior during calls)

### Skills and Extensions

- [Skills System](docs/skills_system.md) -- built-in skills and the modular framework
- [Third-Party Skills](docs/third_party_skills.md) -- creating and publishing custom skills
- [MCP Gateway](docs/mcp_gateway_reference.md) -- Model Context Protocol integration

### Search System

- [Search Overview](docs/search_overview.md) -- architecture, installation, quick start
- [Search Indexing](docs/search_indexing.md) -- building indexes, chunking, embeddings
- [Search Integration](docs/search_integration.md) -- agent integration, skills, HTTP API
- [Search Deployment](docs/search_deployment.md) -- production deployment, pgvector, scaling

### Deployment

- [CLI Guide](docs/cli_guide.md) -- `swaig-test` and `sw-search` command reference
- [Cloud Functions](docs/cloud_functions_guide.md) -- Lambda, Cloud Functions, Azure deployment
- [Bedrock Agent](docs/bedrock_agent.md) -- Amazon Bedrock integration
- [Configuration](docs/configuration.md) -- environment variables, SSL, proxy setup
- [Security](docs/security.md) -- authentication and security model

### Reference

- [API Reference](docs/api_reference.md) -- complete class and method reference
- [Web Service](docs/web_service.md) -- HTTP server and endpoint details
- [Skills Parameter Schema](docs/skills_parameter_schema.md) -- skill parameter definitions

### Tutorials

- [Multi-Agent Tutorial](tutorial/multi_agents/README.md) -- 5-lesson guide from first agent to multi-agent systems
- [Fred Bot Tutorial](tutorial/fred/tutorial/README.md) -- build a Wikipedia AI assistant step-by-step

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SWML_BASIC_AUTH_USER` | Basic auth username (default: auto-generated) |
| `SWML_BASIC_AUTH_PASSWORD` | Basic auth password (default: auto-generated) |
| `SWML_PROXY_URL_BASE` | Base URL when behind a reverse proxy |
| `SWML_SSL_ENABLED` | Enable HTTPS (`true`, `1`, `yes`) |
| `SWML_SSL_CERT_PATH` | Path to SSL certificate |
| `SWML_SSL_KEY_PATH` | Path to SSL private key |
| `SWML_DOMAIN` | Domain for SSL and external URLs |

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run the test suite
pytest

# Run by category
pytest -m unit
pytest -m integration
pytest -m skills

# Coverage
pytest --cov=signalwire_agents --cov-report=html
```

## License

MIT -- see [LICENSE](LICENSE) for details.
