# SignalWire AI Agents SDK - Examples

This directory contains working examples demonstrating the features of the SignalWire AI Agents SDK.

## Setup

```bash
# Install the SDK in development mode (from the repository root)
pip install -e .

# For search examples, install search extras
pip install -e .[search-all]
```

## Running Examples

```bash
# Run any example directly
python examples/simple_agent.py

# Test without running a server
swaig-test examples/simple_agent.py --list-tools
swaig-test examples/simple_agent.py --dump-swml
swaig-test examples/simple_agent.py --exec get_weather --location "New York"
```

## Examples by Category

### Getting Started

| File | Description |
|------|-------------|
| [simple_agent.py](simple_agent.py) | Full-featured agent with POM prompts, SWAIG tools, multilingual support, SIP routing, and LLM parameter tuning |
| [simple_static_agent.py](simple_static_agent.py) | Minimal static agent with no dynamic configuration |
| [simple_dynamic_agent.py](simple_dynamic_agent.py) | Agent with per-request dynamic configuration callback |
| [declarative_agent.py](declarative_agent.py) | Declarative prompt definition using `PROMPT_SECTIONS` class attribute |

### Dynamic Configuration

| File | Description |
|------|-------------|
| [simple_dynamic_enhanced.py](simple_dynamic_enhanced.py) | Dynamic VIP/department/language routing with tier-based voice selection |
| [comprehensive_dynamic_agent.py](comprehensive_dynamic_agent.py) | Full dynamic configuration: tier-based voices, industry prompts, A/B testing, multi-tenant global data |
| [custom_path_agent.py](custom_path_agent.py) | Custom route paths and endpoint configuration |

### Contexts and Steps

| File | Description |
|------|-------------|
| [contexts_demo.py](contexts_demo.py) | Multi-persona sales workflow with context switching (Franklin/Rachael/Dwight), step navigation, and enter fillers |
| [gather_info_demo.py](gather_info_demo.py) | Structured data collection using `set_gather_info()` and `add_gather_question()` in the contexts system |

### Skills

| File | Description |
|------|-------------|
| [skills_demo.py](skills_demo.py) | Loading and configuring built-in skills (datetime, math) |
| [joke_skill_demo.py](joke_skill_demo.py) | DataMap-based joke skill with API integration |
| [joke_agent.py](joke_agent.py) | Joke agent using the joke skill with fallback handling |
| [wikipedia_demo.py](wikipedia_demo.py) | Wikipedia search skill integration |
| [web_search_agent.py](web_search_agent.py) | Google Custom Search API integration via the web_search skill |
| [web_search_multi_instance_demo.py](web_search_multi_instance_demo.py) | Multiple web search skill instances with different configurations |

### Search and Knowledge

| File | Description |
|------|-------------|
| [sigmond_simple.py](sigmond_simple.py) | Simple agent with local `.swsearch` file-based knowledge search |
| [sigmond_native_search.py](sigmond_native_search.py) | Native vector search skill with local search index |
| [sigmond_remote_search.py](sigmond_remote_search.py) | Remote search via HTTP endpoint |
| [pgvector_search_agent.py](pgvector_search_agent.py) | PGVector backend for document search with PostgreSQL |
| [search_with_custom_formatter.py](search_with_custom_formatter.py) | Custom response formatter callback for search results |
| [search_server_standalone.py](search_server_standalone.py) | Standalone search server without an agent |

### DataMap (Serverless Tools)

| File | Description |
|------|-------------|
| [data_map_demo.py](data_map_demo.py) | DataMap builder API for creating server-side tools without webhooks |
| [advanced_datamap_demo.py](advanced_datamap_demo.py) | Advanced DataMap: expressions, foreach loops, multiple webhooks, error handling |

### Datasphere Integration

| File | Description |
|------|-------------|
| [datasphere_multi_instance_demo.py](datasphere_multi_instance_demo.py) | Multiple Datasphere document search instances |
| [datasphere_serverless_demo.py](datasphere_serverless_demo.py) | Serverless Datasphere search using DataMap |
| [datasphere_serverless_env_demo.py](datasphere_serverless_env_demo.py) | Datasphere search with environment variable configuration |
| [datasphere_webhook_env_demo.py](datasphere_webhook_env_demo.py) | Datasphere search via webhook with environment variables |

### Prefab Agents

| File | Description |
|------|-------------|
| [info_gatherer_example.py](info_gatherer_example.py) | InfoGathererAgent prefab for structured data collection |
| [dynamic_info_gatherer_example.py](dynamic_info_gatherer_example.py) | Dynamic InfoGatherer with per-request field configuration |
| [survey_agent_example.py](survey_agent_example.py) | SurveyAgent prefab for conducting surveys |
| [receptionist_agent_example.py](receptionist_agent_example.py) | ReceptionistAgent prefab for call routing |
| [concierge_agent_example.py](concierge_agent_example.py) | ConciergeAgent prefab for multi-department routing |
| [faq_bot_agent.py](faq_bot_agent.py) | FAQ bot prefab with knowledge base |

### Multi-Agent Servers

| File | Description |
|------|-------------|
| [multi_agent_server.py](multi_agent_server.py) | Multiple agents on one server: healthcare, finance, retail, plus InfoGatherer prefabs |
| [multi_endpoint_agent.py](multi_endpoint_agent.py) | Single agent class serving multiple endpoints |

### MCP Gateway

| File | Description |
|------|-------------|
| [mcp_gateway_demo.py](mcp_gateway_demo.py) | Connect to MCP (Model Context Protocol) servers through the mcp_gateway skill |

### LLM Parameters

| File | Description |
|------|-------------|
| [llm_params_demo.py](llm_params_demo.py) | LLM parameter tuning with three persona examples (customer service, creative, technical) |

### SWAIG Features

| File | Description |
|------|-------------|
| [swaig_features_agent.py](swaig_features_agent.py) | Advanced SWAIG function features and configuration |
| [record_call_example.py](record_call_example.py) | Call recording configuration |
| [room_and_sip_example.py](room_and_sip_example.py) | Room management and SIP integration |
| [tap_example.py](tap_example.py) | Call tap/monitoring setup |

### SWML Services

| File | Description |
|------|-------------|
| [basic_swml_service.py](basic_swml_service.py) | Low-level SWML document creation using SWMLService |
| [swml_service_example.py](swml_service_example.py) | SWML service with custom verbs and document structure |
| [dynamic_swml_service.py](dynamic_swml_service.py) | Dynamic SWML generation based on request parameters |
| [swml_service_routing_example.py](swml_service_routing_example.py) | SWML service with route-based document selection |
| [auto_vivified_example.py](auto_vivified_example.py) | Auto-vivified SWML document construction |

### Amazon Bedrock

| File | Description |
|------|-------------|
| [bedrock_with_skills.py](bedrock_with_skills.py) | BedrockAgent with skills, tools, and full configuration |
| [bedrock_agent_run.py](bedrock_agent_run.py) | Basic BedrockAgent setup and run |
| [bedrock_agent_test.py](bedrock_agent_test.py) | BedrockAgent testing patterns |
| [bedrock_server_test.py](bedrock_server_test.py) | BedrockAgent server deployment test |

### Deployment

| File | Description |
|------|-------------|
| [lambda_agent.py](lambda_agent.py) | AWS Lambda deployment example |
| [kubernetes_ready_agent.py](kubernetes_ready_agent.py) | Kubernetes-ready agent with health checks |
| [Dockerfile.flexible](Dockerfile.flexible) | Multi-stage Dockerfile for any agent |
| [Dockerfile.k8s](Dockerfile.k8s) | Kubernetes-optimized Dockerfile |
| [k8s-deployment.yaml](k8s-deployment.yaml) | Kubernetes deployment manifest |

## Authentication

Agents auto-generate credentials on startup. To set fixed credentials:

```bash
export SWML_BASIC_AUTH_USER=myuser
export SWML_BASIC_AUTH_PASSWORD=mypassword
python examples/simple_agent.py
```

## Testing with swaig-test

The `swaig-test` CLI tool lets you test any example without running a server:

```bash
# List all tools an agent exposes
swaig-test examples/simple_agent.py --list-tools

# Generate the SWML document
swaig-test examples/simple_agent.py --dump-swml

# Execute a specific tool
swaig-test examples/simple_agent.py --exec get_weather --location "San Francisco"

# Test in a serverless environment
swaig-test examples/lambda_agent.py --simulate-serverless lambda --dump-swml

# Multi-agent files - select by class or route
swaig-test examples/multi_agent_server.py --agent-class HealthcareAgent --dump-swml
swaig-test examples/multi_agent_server.py --route /finance --list-tools
```

See [CLI Guide](../docs/cli_guide.md) for full documentation.
