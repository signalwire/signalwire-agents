# LLM Parameters Guide

This guide explains how to customize Language Model (LLM) parameters in SignalWire AI Agents to fine-tune the AI's behavior for your specific use case.

## Overview

SignalWire AI Agents SDK provides methods to customize LLM parameters for both the main prompt and post-prompt, allowing precise control over the AI's response characteristics.

**Important:** The SDK passes parameters through to the SignalWire server without validation. Model-specific parameters are validated and handled by the server based on the target model's capabilities and requirements. Invalid parameters for the selected model will be handled or ignored by the server.

## Available Methods

### set_prompt_llm_params(**params)

Sets LLM parameters for the main agent prompt. Accepts any parameters that will be passed to the server.

```python
agent.set_prompt_llm_params(
    temperature=0.7,
    top_p=0.9,
    barge_confidence=0.6,
    presence_penalty=0.0,
    frequency_penalty=0.0
)
```

### set_post_prompt_llm_params(**params)

Sets LLM parameters for the post-prompt (conversation summary). Accepts any parameters that will be passed to the server.

```python
agent.set_post_prompt_llm_params(
    temperature=0.3,
    top_p=0.95,
    presence_penalty=0.0,
    frequency_penalty=0.0
)
```

Note: barge_confidence is not applicable to post-prompt as interruption doesn't apply to summaries.

## Common Parameter Descriptions

These are commonly used parameters, but any parameter accepted by your model can be used. The actual ranges and defaults are model-specific and handled by the server.

### temperature
Controls the randomness of the AI's responses.
- **Lower values (e.g., 0.0-0.3)**: More deterministic, focused, and consistent responses
- **Medium values (e.g., 0.4-0.7)**: Balanced creativity and consistency
- **Higher values (e.g., 0.8+)**: More creative, diverse, and unpredictable responses

### top_p
Nucleus sampling parameter that controls the cumulative probability of token selection.
- **Lower values (e.g., 0.1-0.5)**: Only considers the most likely tokens
- **Medium values (e.g., 0.6-0.9)**: Balanced token selection
- **Higher values (e.g., 0.95-1.0)**: Considers a wider range of tokens

### barge_confidence
ASR (Automatic Speech Recognition) confidence threshold to interrupt the AI while it's speaking (main prompt only).
- **Lower values (e.g., 0.0-0.4)**: Easier to interrupt, more sensitive to user speech
- **Medium values (e.g., 0.5-0.7)**: Balanced interruption sensitivity
- **Higher values (e.g., 0.8-1.0)**: Harder to interrupt, requires clear user speech

### presence_penalty
Topic diversity control. Penalizes tokens based on whether they appear in the conversation so far.
- **Negative values**: Encourages repetition of topics
- **Zero**: No penalty
- **Positive values**: Discourages repetition, encourages new topics

### frequency_penalty
Repetition control. Penalizes tokens based on their frequency in the conversation.
- **Negative values**: Encourages repetition of specific words
- **Zero**: No penalty
- **Positive values**: Discourages word repetition, encourages vocabulary variety

**Note:** No default values are sent unless explicitly set using the methods above. The server will apply model-appropriate defaults if parameters are not specified.

## Use Case Examples

### Customer Service Agent
```python
class CustomerServiceAgent(AgentBase):
    def __init__(self):
        super().__init__(name="customer-service", route="/support")
        
        self.prompt_add_section("Role", "You are a professional customer service representative.")
        
        # Consistent, helpful responses
        self.set_prompt_llm_params(
            temperature=0.3,        # Low randomness for consistency
            top_p=0.9,             # Focused token selection
            barge_confidence=0.6,  # Moderate interruption threshold (default 0.0 is too easy)
            presence_penalty=0.1,  # Slight penalty to avoid repetition
            frequency_penalty=0.1  # Encourage varied language
        )
```

### Creative Writing Assistant
```python
class CreativeWritingAgent(AgentBase):
    def __init__(self):
        super().__init__(name="creative-writer", route="/writer")
        
        self.prompt_add_section("Role", "You are a creative writing assistant.")
        
        # Creative, diverse responses
        self.set_prompt_llm_params(
            temperature=0.8,        # High randomness for creativity
            top_p=0.95,            # Wide token selection
            barge_confidence=0.3,  # Easy to interrupt for collaboration (but not default 0.0)
            presence_penalty=-0.1, # Allow topic revisiting
            frequency_penalty=0.3  # Encourage vocabulary diversity
        )
```

### Technical Documentation Bot
```python
class TechnicalDocsAgent(AgentBase):
    def __init__(self):
        super().__init__(name="tech-docs", route="/docs")
        
        self.prompt_add_section("Role", "You are a technical documentation assistant.")
        
        # Precise, accurate responses
        self.set_prompt_llm_params(
            temperature=0.2,        # Very low randomness
            top_p=0.8,             # More focused token selection
            barge_confidence=0.8,  # Hard to interrupt - let it finish
            presence_penalty=0.0,  # Neutral on repetition
            frequency_penalty=0.2  # Some vocabulary variety
        )
        
        # Even more focused for summaries
        self.set_post_prompt_llm_params(
            temperature=0.1       # Extremely consistent
        )
```

### Legal Advisor Bot
```python
class LegalAdvisorAgent(AgentBase):
    def __init__(self):
        super().__init__(name="legal-advisor", route="/legal")
        
        self.prompt_add_section("Role", "You are a legal information assistant.")
        self.prompt_add_section("Disclaimer", "Always remind users to consult a real attorney.")
        
        # Cautious, precise responses
        self.set_prompt_llm_params(
            temperature=0.2,        # Very consistent
            top_p=0.85,            # Focused selection
            barge_confidence=0.9,  # Very hard to interrupt - legal accuracy important
            presence_penalty=0.0,  # Allow legal term repetition
            frequency_penalty=0.0  # Legal language often repeats
        )
```

## Best Practices

### 1. Start with Defaults
Begin with the default values and adjust based on observed behavior.

### 2. Test Incrementally
Make small adjustments and test thoroughly to understand the impact.

### 3. Consider the Use Case
- **Customer Service**: Low temperature (0.2-0.4), moderate barge_confidence (0.6-0.7)
- **Creative Tasks**: Higher temperature (0.7-0.9), low barge_confidence (0.4-0.6)
- **Technical/Legal**: Very low temperature (0.1-0.3), high barge_confidence (0.8-0.9)
- **General Assistant**: Medium temperature (0.5-0.7), medium barge_confidence (0.6-0.7)

### 4. Match Post-Prompt Parameters
Post-prompt parameters should typically be lower temperature than main prompt for consistent summaries.

### 5. Monitor Barge Confidence Levels
- Too high: Users have difficulty interrupting the AI
- Too low: AI gets interrupted too easily by background noise

## Parameter Interactions

### Temperature + Top-p
These parameters work together to control randomness:
- Low temperature + Low top_p = Very focused responses
- High temperature + High top_p = Maximum creativity
- Low temperature + High top_p = Consistent but with fallback options
- High temperature + Low top_p = Creative within constraints

### Penalty Parameters
Presence and frequency penalties can be used together:
- Both positive: Strong encouragement for variety
- Both negative: Strong encouragement for repetition
- Mixed: Fine-tuned control over specific repetition patterns

## Troubleshooting

### AI is too repetitive
- Increase `presence_penalty` (try 0.3-0.6)
- Increase `frequency_penalty` (try 0.3-0.6)
- Slightly increase `temperature`

### AI is too random/inconsistent
- Decrease `temperature` (try 0.2-0.4)
- Decrease `top_p` (try 0.7-0.85)

### AI gets interrupted too easily
- Increase `barge_confidence` threshold
- Check for background noise in the environment
- Consider the user's speaking clarity

### Users can't interrupt the AI
- Decrease `barge_confidence` threshold
- Train users to speak more clearly when interrupting
- Consider the use case (e.g., legal/medical may need higher thresholds)

## Parameter Behavior

**No Default Values:** The SDK does not send any LLM parameters unless explicitly set using `set_prompt_llm_params()` or `set_post_prompt_llm_params()`. When parameters are not specified, the SignalWire server will apply appropriate defaults based on the model being used.

**Server-Side Validation:** All parameter validation is handled by the SignalWire server. The SDK accepts any parameters and passes them through without modification. This allows:
- Use of model-specific parameters without SDK updates
- Forward compatibility with new models and parameters
- Server-side optimization based on model capabilities

**Partial Configuration:** You can set only the parameters you want to customize. For example:
```python
# Only set temperature, let server handle other parameters
agent.set_prompt_llm_params(temperature=0.7)

# Or set multiple specific parameters
agent.set_prompt_llm_params(
    temperature=0.5,
    barge_confidence=0.6
)
```