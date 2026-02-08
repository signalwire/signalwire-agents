# Claude Skills Loader

Load Claude Code-style SKILL.md files as SignalWire agent tools.

## Overview

This skill bridges Claude Code skills into SignalWire AI agents. It parses Claude skill directories (each containing a `SKILL.md` file) and makes them available as SWAIG tools that the AI can invoke.

## Claude Skill Format

Claude skills use a simple markdown format with YAML frontmatter:

```yaml
---
name: explain-code
description: Use when explaining how code works or when user asks "how does this work?"
---

When explaining code, always include:
1. Start with an analogy
2. Draw a diagram
3. Walk through step-by-step

Use $ARGUMENTS for context passed to this skill.
```

## Usage

```python
from signalwire_agents import AgentBase

agent = AgentBase(name="my-agent")

# Load Claude skills from a directory
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",  # Path to Claude skills directory
})
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `skills_path` | string | Yes | - | Path to directory containing Claude skill folders |
| `include` | array | No | `["*"]` | Glob patterns for skills to include |
| `exclude` | array | No | `[]` | Glob patterns for skills to exclude |
| `tool_prefix` | string | No | `"claude_"` | Prefix for generated tool names (use `""` for no prefix) |
| `prompt_title` | string | No | `"Claude Skills"` | Title for the prompt section |
| `prompt_intro` | string | No | See below | Intro text for prompt section |
| `skill_descriptions` | object | No | `{}` | Override descriptions for specific skills |
| `response_prefix` | string | No | `""` | Text to prepend to skill results |
| `response_postfix` | string | No | `""` | Text to append to skill results |
| `swaig_fields` | object | No | `{}` | Extra SWAIG fields (fillers, wait_file, etc.) |

## How It Works

1. **Discovery**: Scans `skills_path` for directories containing `SKILL.md` files
2. **Parsing**: Extracts YAML frontmatter (name, description) and markdown body
3. **Prompt Injection**: Adds a prompt section telling the AI when to use each skill
4. **Tool Registration**: Creates a SWAIG tool for each skill (e.g., `claude_explain_code`)
5. **Execution**: When called, returns the skill's instructions with arguments substituted

## Argument Substitution

Claude skills support argument placeholders:

- `$ARGUMENTS` - Full arguments string
- `$0`, `$1`, `$2`... - Positional arguments (space-separated)
- `$ARGUMENTS[0]`, `$ARGUMENTS[1]`... - Same as above

Example skill:
```yaml
---
name: migrate-component
description: Migrate a component between frameworks
---

Migrate the $0 component from $1 to $2.
Preserve all existing behavior.
```

When called with arguments `"Button React Vue"`, expands to:
```
Migrate the Button component from React to Vue.
Preserve all existing behavior.
```

## Customizing Descriptions

Override skill descriptions when loading:

```python
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "skill_descriptions": {
        "explain-code": "Use when user asks how something works",
        "commit": "Use when user wants to create a git commit",
    }
})
```

Priority: `skill_descriptions` override > SKILL.md `description` > skill name

## Filtering Skills

Include or exclude specific skills:

```python
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "include": ["explain-*", "code-*"],  # Only these patterns
    "exclude": ["*-internal"],            # Skip these
})
```

## Tool Prefix

By default, tools are prefixed with `claude_`. You can change or remove this:

```python
# Custom prefix
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "tool_prefix": "skill_",  # Results in skill_explain_code
})

# No prefix
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "tool_prefix": "",  # Results in explain_code
})
```

## SWAIG Fields

Add fillers, wait files, or other SWAIG fields to all generated tools:

```python
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "swaig_fields": {
        "fillers": {"en": ["Just a moment...", "Let me check..."]},
        "wait_file": "https://example.com/hold.mp3",
        "wait_file_loops": 3
    }
})
```

## Response Wrapping

Wrap skill results with prefix and postfix text to provide context or instructions to the AI:

```python
agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "response_prefix": "Use the following skill instructions to help the user:",
    "response_postfix": "Remember to be concise and helpful in your response."
})
```

This wraps all skill results:
```
Use the following skill instructions to help the user:

[SKILL.md content or section content here]

Remember to be concise and helpful in your response.
```

## Generated Tools

Each Claude skill becomes a SWAIG tool named `{prefix}{skill_name}` (default prefix: `claude_`):

| Claude Skill | Default Tool Name | With `tool_prefix=""` |
|--------------|-------------------|----------------------|
| `explain-code` | `claude_explain_code` | `explain_code` |
| `review-pr` | `claude_review_pr` | `review_pr` |
| `commit` | `claude_commit` | `commit` |

## Prompt Section

The skill automatically adds a prompt section like:

```
## Claude Skills

You have access to specialized skills. Call the appropriate tool when the user's question matches:

- claude_explain_code: Use when explaining how code works
- claude_review_pr: Use when reviewing pull requests
- claude_commit: Use when creating git commits
```

## Supporting Files (Progressive Disclosure)

Skills can include additional markdown files that are loaded on-demand. This enables progressive disclosure - the SKILL.md content goes into the prompt as a table of contents, while supporting files are loaded only when the AI calls the tool with a specific section.

### Directory Structure

```
my-skill/
├── SKILL.md              # TOC - goes in prompt
├── reference.md          # Supporting file
├── examples.md           # Supporting file
├── references/
│   ├── api.md            # Nested supporting file
│   └── errors.md
└── templates/
    └── template.md
```

### How It Works

1. **Discovery**: All `.md` files in the skill directory (recursively) are discovered
2. **Prompt Section**: SKILL.md body becomes a TOC in the agent's prompt
3. **Tool Registration**: ONE tool per skill with optional `section` enum parameter
4. **Enum Values**: All discovered .md files (e.g., `["reference", "examples", "references/api", "references/errors", "templates/template"]`)
5. **Execution**: When called with a section, return that file's content

### Example SKILL.md with Sections

```yaml
---
name: explain-code
description: Explains code with diagrams and analogies
---

When explaining code, always include:
1. Start with an analogy
2. Draw a diagram
3. Walk through step-by-step

Use the reference sections for detailed guidance.
```

### Generated Tool

When sections exist, the tool has a `section` enum parameter:

```json
{
  "function": "claude_explain_code",
  "parameters": {
    "type": "object",
    "properties": {
      "section": {
        "type": "string",
        "description": "Which reference section to load",
        "enum": ["examples", "reference", "references/api", "references/errors"]
      },
      "arguments": {
        "type": "string",
        "description": "Arguments or context to pass to the skill"
      }
    }
  }
}
```

### Prompt Section

The agent's prompt includes the SKILL.md body plus available sections:

```
## explain-code

When explaining code, always include:
1. Start with an analogy
2. Draw a diagram
3. Walk through step-by-step

Use the reference sections for detailed guidance.

Available reference sections: examples, reference, references/api, references/errors
Call claude_explain_code(section="<name>") to load a section.
```

### Benefits

- **Reduced prompt size**: Only SKILL.md goes in the prompt; supporting files load on-demand
- **Better organization**: Split large skill documentation into logical sections
- **Nested structure**: Organize files in subdirectories for complex skills

## Limitations

This skill does **not** support:

- `!`command`` shell injection (security concern)
- `context: fork` subagent execution
- `allowed-tools` restrictions

## Example

Directory structure:
```
~/.claude/skills/
├── explain-code/
│   ├── SKILL.md
│   ├── examples.md
│   └── references/
│       └── patterns.md
├── review-pr/
│   └── SKILL.md
└── commit/
    └── SKILL.md
```

Agent code:
```python
from signalwire_agents import AgentBase

agent = AgentBase(name="code-assistant")

agent.add_skill("claude_skills", {
    "skills_path": "~/.claude/skills",
    "prompt_intro": "You have coding assistance tools available:",
})

if __name__ == "__main__":
    agent.serve()
```

The AI will now have access to `claude_explain_code`, `claude_review_pr`, and `claude_commit` tools, and will know when to use them based on user questions.
