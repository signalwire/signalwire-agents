# Changelog

## [0.1.49] - 2025-10-10

- Version bump

## [0.1.48] - 2025-10-02

- Version bump

## [0.1.47] - 2025-09-12

- Version bump

## [0.1.46] - 2025-09-04

- Version bump

## [0.1.45] - 2025-09-04

- Version bump

## [0.1.44] - 2025-08-12

- Version bump

## [0.1.43] - 2025-08-11

- Version bump

## [0.1.42] - 2025-08-07

- Version bump

## [0.1.41] - 2025-07-22

- Version bump

## [0.1.40] - 2025-07-22

- Version bump

## [0.1.39] - 2025-07-17

- Version bump

## [0.1.38] - 2025-07-07

- Version bump

## [0.1.37] - 2025-07-07

- Version bump

## [Unreleased]

### Added
- LLM parameter customization support via `set_prompt_llm_params()` and `set_post_prompt_llm_params()` methods
- Fine-grained control over AI behavior with temperature, top_p, confidence, presence_penalty, and frequency_penalty
- pgvector backend support for search system with PostgreSQL vector database
- `--overwrite` flag for search index building to replace existing collections
- SearchService now supports both SQLite and pgvector backends

### Documentation
- Added comprehensive LLM Parameters Guide (docs/llm_parameters.md)
- Updated README with LLM parameter examples
- Updated agent_guide.md with LLM parameter configuration section
- Updated API reference with new LLM parameter methods
- Enhanced search system documentation with pgvector information

## [0.1.36] - 2025-07-01

- Version bump

## [0.1.35] - 2025-06-26

- Version bump

## [0.1.34] - 2025-06-24

- Version bump

## [0.1.33] - 2025-06-24

- Version bump

## [0.1.32] - 2025-06-24

- Version bump

## [0.1.31] - 2025-06-24

- Version bump

## [0.1.30] - 2025-06-24

- Version bump

## [0.1.29] - 2025-06-23

- Version bump

## [0.1.28] - 2025-06-23

- Version bump

## [0.1.27] - 2025-06-20

- Version bump

## [0.1.26] - 2025-06-20

- Version bump

## [0.1.25] - 2025-06-19

- Version bump

## [0.1.24] - 2025-06-19

- Version bump

## [0.1.23] - 2025-06-17

- Version bump

## [0.1.22] - 2025-06-17

- Version bump

## [0.1.21] - 2025-06-12

- Version bump

## [0.1.20] - 2025-06-10

- Version bump

## [0.1.19] - 2025-06-09

- Version bump

## [0.1.18] - 2025-06-09

- Version bump

## [0.1.17] - 2025-06-02

- Version bump

## [0.1.16] - 2025-06-02

- Version bump

## [0.1.15] - 2025-05-30

- Version bump

## [0.1.14] - 2025-05-29

- Version bump

## [0.1.13] - 2025-05-29

- Version bump

## [0.1.12] - 2025-05-28

- Version bump

## [0.1.11] - 2025-05-28

- Version bump

## [0.1.10] - 2025-05-28

- Version bump

## [0.1.9] - 2025-05-24

- Version bump

## [0.1.8] - 2025-05-23

- Version bump

## [0.1.7] - 2025-05-17

- Version bump

## [0.1.6] - 2025-05-16

- Version bump

## [0.1.5] - 2025-05-16

- Version bump

## [0.1.4] - 2025-05-16

- Version bump

## [Unreleased]

### Added

- **SIP Routing:** Added support for routing SIP requests based on username
  - `SWMLService.extract_sip_username()` method for extracting SIP usernames from request bodies
  - `register_routing_callback()` method for SWMLService to handle custom routing
  - `enable_sip_routing()` method for AgentBase to automatically handle SIP requests
  - `register_sip_username()` method for AgentBase to register custom SIP usernames
  - `setup_sip_routing()` method for AgentServer to enable centralized SIP routing
  - Added global `/sip` endpoint for both SWMLService and AgentBase when SIP routing is enabled
  - Example: Added SIP routing to simple_agent.py
  - Documentation: Added docs/sip_routing.md with full documentation

### Fixed

- Fixed the `sleep` verb auto-vivification to properly handle direct integer values

### Changed

- **Dynamic Method Generation:** Implemented auto-vivification in SWMLService class for more intuitive verb usage
  - Now you can use `service.play(url="say:Hello")` instead of `service.add_verb("play", {"url": "say:Hello"})`
  - All verbs from the schema are automatically available as methods
  - Special handling for the `sleep` verb which takes a direct integer value

## [0.1.2] - 2023-04-15

### Added

- Initial release of the SignalWire AI Agent SDK
- Base classes for building and serving AI Agents
- SWML document creation and manipulation
- Web server for handling agent requests
- Documentation and examples 