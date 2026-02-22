# Lesson 5: Creating Custom Functions

Let's make Fred more engaging by adding a custom function that shares fun facts about Wikipedia itself! This lesson teaches you how to create SWAIG functions.

## Table of Contents

1. [Understanding SWAIG Functions](#understanding-swaig-functions)
2. [Creating the Fun Fact Function](#creating-the-fun-fact-function)
3. [The @tool Decorator](#the-tool-decorator)
4. [Function Implementation](#function-implementation)
5. [Returning Results](#returning-results)

---

## Understanding SWAIG Functions

SWAIG (SignalWire AI Gateway) functions are tools that your AI agent can call during conversations. They enable your agent to perform actions beyond just talking.

### What is a SWAIG Function?

**A SWAIG function:**
- Is callable by the AI during conversations
- Receives parameters from the AI
- Performs an action (API call, calculation, etc.)
- Returns a result the AI can use

### Function Flow

```
User asks question → AI decides to call function → Function executes → Result returned to AI → AI responds to user
```

## Creating the Fun Fact Function

Let's add a function that shares random Wikipedia facts. This will make Fred more entertaining!

### Step 1: Add the Function Decorator

Add this inside Fred's `__init__` method, after the skill configuration:

```python
        # Add a fun fact function
        @self.tool(
            name="share_fun_fact",
            description="Share a fun fact about Wikipedia itself",
            parameters={}
        )
        def share_fun_fact(args, raw_data):
            import random
            facts = [
                "Wikipedia has over 6 million articles in English alone!",
                "Wikipedia is available in more than 300 languages!",
                "Wikipedia was launched on January 15, 2001!",
                "The most edited Wikipedia page is about George W. Bush!",
                "Wikipedia is the 7th most visited website in the world!"
            ]
            fact = random.choice(facts)
            return SwaigFunctionResult(f"Here's a fun Wikipedia fact: {fact}")
```

### Understanding the Code

Let's break down each part:

**The Decorator:**
```python
@self.tool(
    name="share_fun_fact",          # Function name the AI will use
    description="Share a fun...",   # Helps AI know when to use it
    parameters={}                   # No parameters needed
)
```

**The Function Signature:**
```python
def share_fun_fact(args, raw_data):
    # args: Dictionary of parameters passed by AI
    # raw_data: Full request data (call_id, metadata, etc.)
```

**The Implementation:**
```python
import random                       # Import inside function is fine
facts = [...]                       # List of facts
fact = random.choice(facts)         # Pick random fact
return SwaigFunctionResult(f"...")  # Return formatted result
```

## The @tool Decorator

The `@self.tool()` decorator is Fred's way of registering SWAIG functions. Let's explore its parameters:

### Decorator Parameters

```python
@self.tool(
    name="function_name",           # Required: Unique function identifier
    description="What it does",     # Required: AI uses this to decide when to call
    parameters={                    # Required: Parameter definitions
        "param_name": {
            "type": "string",       # Data type: string, number, boolean
            "description": "...",   # What the parameter is for
            "enum": ["opt1", "opt2"] # Optional: Allowed values
        }
    }
)
```

### Example with Parameters

Here's an example function that takes parameters:

```python
@self.tool(
    name="search_by_category",
    description="Search Wikipedia articles in a specific category",
    parameters={
        "category": {
            "type": "string",
            "description": "Category to search in",
            "enum": ["science", "history", "geography", "people"]
        },
        "limit": {
            "type": "number",
            "description": "Maximum results to return",
            "default": 3
        }
    }
)
def search_by_category(args, raw_data):
    category = args.get("category", "general")
    limit = args.get("limit", 3)
    # Implementation here...
    return SwaigFunctionResult(f"Found {limit} articles in {category}")
```

## Function Implementation

Let's create a more sophisticated version of our fun fact function:

### Step 2: Enhanced Fun Fact Function

Replace the simple version with this enhanced one:

```python
        # Enhanced fun fact function with categories
        @self.tool(
            name="share_fun_fact",
            description="Share an interesting fact about Wikipedia itself",
            parameters={
                "category": {
                    "type": "string",
                    "description": "Type of fact to share",
                    "enum": ["statistics", "history", "records", "random"]
                }
            }
        )
        def share_fun_fact(args, raw_data):
            import random
            
            # Get the requested category
            category = args.get("category", "random")
            
            # Define facts by category
            facts = {
                "statistics": [
                    "Wikipedia has over 6 million articles in English alone!",
                    "Wikipedia is available in more than 300 languages!",
                    "Wikipedia receives over 18 billion page views per month!",
                    "There are over 100,000 active Wikipedia contributors!"
                ],
                "history": [
                    "Wikipedia was launched on January 15, 2001!",
                    "The first Wikipedia article was about the letter 'U'!",
                    "Wikipedia's name comes from 'wiki' (Hawaiian for 'quick') and 'encyclopedia'!",
                    "Jimmy Wales and Larry Sanger founded Wikipedia!"
                ],
                "records": [
                    "The most edited Wikipedia page is about George W. Bush!",
                    "The longest Wikipedia article is about California Proposition 8!",
                    "Wikipedia is the 7th most visited website in the world!",
                    "The Wikipedia article on 'List of Pokemon' is one of the most viewed!"
                ],
                "random": []  # Will be filled with all facts
            }
            
            # Combine all facts for random selection
            all_facts = []
            for fact_list in facts.values():
                if fact_list:  # Skip empty random list
                    all_facts.extend(fact_list)
            facts["random"] = all_facts
            
            # Select appropriate fact
            fact_list = facts.get(category, facts["random"])
            if not fact_list:
                return SwaigFunctionResult("I don't have any facts in that category.")
            
            fact = random.choice(fact_list)
            
            # Add category context to response
            if category != "random":
                return SwaigFunctionResult(f"Here's a {category} fact about Wikipedia: {fact}")
            else:
                return SwaigFunctionResult(f"Here's a fun Wikipedia fact: {fact}")
```

### Best Practices for Function Implementation

1. **Parameter Validation**
   ```python
   category = args.get("category", "random")  # Always provide defaults
   ```

2. **Error Handling**
   ```python
   if not fact_list:
       return SwaigFunctionResult("I don't have any facts in that category.")
   ```

3. **Clear Responses**
   ```python
   # Include context in response
   return SwaigFunctionResult(f"Here's a {category} fact: {fact}")
   ```

## Returning Results

SWAIG functions must return a `SwaigFunctionResult` object. This class provides several useful features:

### Basic Result

```python
return SwaigFunctionResult("Simple text response")
```

### Result with Actions

```python
result = SwaigFunctionResult("Playing background music")
result.add_action("play_audio", {"url": "https://example.com/music.mp3"})
return result
```

### Multiple Actions

```python
result = SwaigFunctionResult("Setting up the call")
result.add_actions([
    {"play_audio": {"url": "https://example.com/intro.mp3"}},
    {"set_var": {"name": "call_started", "value": True}}
])
return result
```

### Available Actions

Common SWAIG actions include:
- `play_audio` - Play background audio
- `stop_audio` - Stop background audio
- `transfer` - Transfer the call
- `hangup` - End the call
- `set_var` - Set a variable
- `play_tts` - Play text-to-speech

## Testing Custom Functions

Let's update our main function to highlight both capabilities:

```python
def main():
    """Run Fred the Wiki Bot"""
    print("=" * 60)
    print("🤖 Fred - The Wikipedia Knowledge Bot")
    print("=" * 60)
    print()
    print("Fred is a friendly assistant who loves searching Wikipedia!")
    print("He can help you learn about almost any topic.")
    print()
    print("Example questions you can ask Fred:")
    print("  • 'Tell me about Albert Einstein'")
    print("  • 'What is quantum physics?'")
    print("  • 'Who was Marie Curie?'")
    print("  • 'Search for information about the solar system'")
    print("  • 'Can you share a fun fact?'")
    print()
    
    # Create and run Fred
    fred = FredTheWikiBot()
    
    # Get auth credentials for display
    username, password = fred.get_basic_auth_credentials()
    
    print(f"Fred is available at: http://localhost:3000/fred")
    print(f"Basic Auth: {username}:{password}")
    print()
    print("Fred's capabilities:")
    print("  ✓ Wikipedia search (via skill)")
    print("  ✓ Fun facts about Wikipedia (custom function)")
    print()
    print("Starting Fred... Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        fred.run()
    except KeyboardInterrupt:
        print("\n👋 Fred says goodbye! Thanks for learning with me!")
```

## Complete Function Reference

Here's the complete fun fact function with all features:

```python
# Inside __init__ method:

# Add a fun fact function
@self.tool(
    name="share_fun_fact",
    description="Share an interesting fact about Wikipedia itself",
    parameters={
        "category": {
            "type": "string",
            "description": "Type of fact to share",
            "enum": ["statistics", "history", "records", "random"]
        }
    }
)
def share_fun_fact(args, raw_data):
    """
    Share interesting facts about Wikipedia
    
    Args:
        args: Dictionary with 'category' parameter
        raw_data: Full request context
        
    Returns:
        SwaigFunctionResult with the fact
    """
    import random
    
    # Implementation as shown above...
    # (Full implementation already provided)
```

## Next Steps

Fred is now complete! He can search Wikipedia and share fun facts. Let's learn how to run and test him.

➡️ Continue to [Lesson 6: Running and Testing Fred](06-running-testing.md)

---

**Function Development Checklist:**
- [x] Understood SWAIG function structure
- [x] Created fun fact function
- [x] Added parameter handling
- [x] Implemented proper result returns
- [x] Enhanced with categories

**Fred's Complete Capabilities:**
- ✅ Search Wikipedia (skill-based)
- ✅ Share Wikipedia facts (custom function)
- ✅ Natural conversation flow
- ✅ Friendly personality

---

[← Previous: Wikipedia Skill](04-wikipedia-skill.md) | [Back to Overview](README.md) | [Next: Running & Testing →](06-running-testing.md)