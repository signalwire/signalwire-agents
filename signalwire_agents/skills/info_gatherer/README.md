# Info Gatherer Skill

Guides an AI agent through a configurable series of questions, collecting answers in namespaced `global_data`. Supports multiple instances so different question sets can run independently on the same agent.

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `questions` | list | Yes | List of question objects (see below) |
| `prefix` | str | No | Prefix for tool names and state namespace |
| `completion_message` | str | No | Message shown after all questions are answered |

Each question object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key_name` | str | Yes | Key used to store the answer |
| `question_text` | str | Yes | The question to ask the user |
| `confirm` | bool | No | Whether to require confirmation |

## Single Instance Usage

```python
from signalwire_agents import AgentBase

agent = AgentBase(name="intake", route="/intake")

agent.add_skill("info_gatherer", {
    "questions": [
        {"key_name": "full_name", "question_text": "What is your full name?"},
        {"key_name": "email", "question_text": "What is your email?", "confirm": True},
        {"key_name": "reason", "question_text": "How can we help you today?"},
    ]
})
```

This registers two tools: `start_questions` and `submit_answer`.

## Multiple Instance Usage

Use the `prefix` parameter to run several questionnaires side by side:

```python
agent.add_skill("info_gatherer", {
    "prefix": "intake",
    "questions": [
        {"key_name": "full_name", "question_text": "What is your full name?"},
        {"key_name": "dob", "question_text": "What is your date of birth?", "confirm": True},
    ]
})

agent.add_skill("info_gatherer", {
    "prefix": "medical",
    "questions": [
        {"key_name": "allergies", "question_text": "Do you have any allergies?"},
        {"key_name": "medications", "question_text": "Are you currently taking any medications?"},
    ]
})
```

This registers four tools (`intake_start_questions`, `intake_submit_answer`, `medical_start_questions`, `medical_submit_answer`) with isolated state namespaces (`skill:intake`, `skill:medical`).

## How It Works

1. The agent prompt instructs the AI to call `start_questions` when the user is ready.
2. `start_questions` reads the current question from namespaced `global_data` and returns an instruction for the AI.
3. The AI asks the question, then calls `submit_answer` with the user's response.
4. `submit_answer` stores the answer, advances the index, and returns either the next question instruction or the completion message.
5. All state mutations go through `update_global_data` so the SignalWire platform persists them across turns.
