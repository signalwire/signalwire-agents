# Changelog

## [1.0.9] - 2025-11-30

- Add Azure Functions serverless support with proper URL detection for webhook URLs
- Add Google Cloud Functions serverless support
- Fix Flask header iteration bug in Google Cloud Functions auth check (use .get() instead of iteration)
- Fix Azure Functions auth check to use .get() method for headers
- Improve serverless mixin to properly detect base URL from request for correct SWML webhook URLs

## [1.0.8] - 2025-11-29

- Fix tool definitions in docs to include parameters
- Add sw-agent-init CLI tool and man pages for all CLI tools

## [1.0.7] - 2025-11-27

- Version bump

## [1.0.6] - 2025-11-27

- Update documentation for contexts, datamap, API reference, SWAIG actions, and function results
- Add comprehensive example testing suite (tests/test_examples.py)
- Add static file serving example

## [1.0.5] - 2025-11-26

- Add setuptools version upper bound (<81) to fix compatibility issues

## [1.0.4] - 2025-11-26

- Add call flow verb insertion API for customizing SWML call flow
  - `add_pre_answer_verb()` - Add verbs before answering (ringback, screening, routing)
  - `add_post_answer_verb()` - Add verbs after answer, before AI (welcome messages, disclaimers)
  - `add_post_ai_verb()` - Add verbs after AI ends (cleanup, transfers, logging)
  - `add_answer_verb()` - Configure the answer verb (max_duration, etc.)
  - `clear_pre_answer_verbs()`, `clear_post_answer_verbs()`, `clear_post_ai_verbs()`
- Fix `auto_answer=False` constructor parameter to actually skip the answer verb
- Add validation for pre-answer safe verbs with helpful warnings

## [1.0.3] - 2025-11-24

- Version bump

## [1.0.2] - 2025-11-24

- Version bump

## [1.0.1] - 2025-11-23

- Version bump

## [1.0.0] - 2025-11-22

- Version bump

