# Search Agent Integration Guide

This document covers how to integrate the search system into your agents, configure search behavior, use metadata and tags for filtering, tune search quality, customize response formatting, and deploy multi-collection search architectures. For background on building search indexes, see [search_indexing.md](search_indexing.md). For deployment patterns, see [search_deployment.md](search_deployment.md). For general search system architecture, see [search_overview.md](search_overview.md).

---

## Basic Integration

### Adding the native_vector_search Skill

The `native_vector_search` skill provides search functionality to agents. The simplest integration requires only a tool name, description, and index path:

```python
from signalwire_agents import AgentBase

class DocsAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="DocsAgent",
            route="/docs"
        )

        # Add search skill
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the documentation for information",
            "index_path": "./knowledge.swsearch"
        })
```

The agent automatically exposes this as a SWAIG (SignalWire AI Gateway) function that the LLM can invoke when it determines a search is needed. SWAIG is the platform's AI tool-calling system.

### Configuration Options

#### Required Parameters

**tool_name** (string)
- Name of the SWAIG function exposed to the LLM.
- Should be descriptive: `search_docs`, `search_api`, `search_knowledge`.

**description** (string)
- Tells the LLM when to use this function.
- Be specific: "Search the API documentation for information about endpoints, authentication, and request formats".

#### Search Behavior Parameters

**count** (integer, default: 5)
- Number of results to return.
- More results provide more context but consume more tokens.

```python
{
    "count": 3  # Return top 3 results
}
```

**distance_threshold** (float, default: 0.5)
- Minimum similarity score (0.0 to 1.0).
- Lower values produce stricter matching; higher values are more permissive.

```python
{
    "distance_threshold": 0.4  # Only results with similarity > 0.4
}
```

**tags** (list, optional)
- Filter results by tags.
- Only chunks with these tags are returned.

```python
{
    "tags": ["api", "reference"]  # Only API reference chunks
}
```

#### User Experience Parameters

**no_results_message** (string, optional)
- Message returned when no results are found.
- Use `{query}` as a placeholder for the search query.

```python
{
    "no_results_message": "I couldn't find information about '{query}' in the documentation. Try rephrasing your question."
}
```

**max_content_length** (integer, default: 32768)
- Maximum total characters in the search response.
- Prevents response truncation.
- Budget is distributed across results.

```python
{
    "max_content_length": 16384  # 16KB total
}
```

**response_prefix** (string, optional)
- Text prepended to the search response.

**response_postfix** (string, optional)
- Text appended to the search response.

```python
{
    "response_prefix": "Based on the documentation, here's what I found:",
    "response_postfix": "Would you like me to search for more specific information?"
}
```

**swaig_fields** (dict, optional)
- SWAIG-specific configuration, including function fillers for better voice UX.

```python
{
    "swaig_fields": {
        "fillers": {
            "en-US": [
                "Let me search the documentation...",
                "I'm looking through the docs...",
                "Searching for that information..."
            ]
        }
    }
}
```

#### NLP Backend Selection

Choose between NLTK (fast) and spaCy (better quality) for query processing:

```python
# Fast NLTK processing (default)
{
    "nlp_backend": "nltk"  # ~50-100ms query processing
}

# Better quality spaCy processing
{
    "nlp_backend": "spacy"  # ~150-300ms query processing, requires model download
}
```

#### Model Selection

When using `build_index: True` for auto-building indexes:

```python
{
    "model_name": "mini"   # Fast, 5x faster, good for most use cases
    # "model_name": "base"  # Balanced, better quality
    # "model_name": "large" # Best quality
}
```

See [Search Indexing](search_indexing.md) for detailed model comparisons.

### Backend Selection (SQLite vs pgvector)

The search system supports multiple storage backends. Choose one per skill instance.

**Option 1: Local .swsearch file (SQLite)**

```python
{
    "index_path": "./knowledge.swsearch"
}
```

Best for single-agent deployments, development, and small to medium datasets. Portable single-file storage.

**Option 2: Remote search server**

```python
{
    "remote_url": "http://localhost:8001",
    "index_name": "docs"
}
```

Best for centralized index management and lower per-agent memory usage.

**Option 3: pgvector database**

```python
{
    "backend": "pgvector",
    "connection_string": "postgresql://user:pass@localhost:5432/db",
    "collection_name": "docs",
    "model_name": "mini"  # Must match model used during indexing
}
```

Best for production deployments, multi-agent systems, and large datasets. Supports concurrent access and real-time updates.

#### Backend Comparison

| Feature | SQLite | pgvector |
|---------|--------|----------|
| Setup complexity | None | Requires PostgreSQL |
| Scalability | Limited | Excellent |
| Concurrent access | Poor | Excellent |
| Update capability | Rebuild required | Real-time |
| Performance (small datasets) | Excellent | Good |
| Performance (large datasets) | Poor | Excellent |
| Deployment | File copy | Database connection |
| Multi-agent sharing | Separate copies | Shared knowledge |

### Complete Configuration Example

A fully configured search skill using pgvector:

```python
class ProductAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="ProductAgent",
            route="/product"
        )

        # Configure search with all options
        self.add_skill("native_vector_search", {
            # Identity
            "tool_name": "search_product_docs",
            "description": "Search comprehensive product documentation including features, configuration, troubleshooting, and API references",

            # Backend (pgvector)
            "backend": "pgvector",
            "connection_string": os.getenv("PGVECTOR_CONNECTION"),
            "collection_name": "product_docs",
            "model_name": "mini",

            # Search behavior
            "count": 5,
            "distance_threshold": 0.4,

            # User experience
            "no_results_message": "I couldn't find information about '{query}' in our documentation. Could you rephrase or ask about a different topic?",
            "max_content_length": 32768,

            # SWAIG configuration
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me check our documentation...",
                        "I'm searching for that information...",
                        "Looking through the product docs...",
                        "One moment while I find that..."
                    ]
                }
            }
        })
```

---

## Search Behavior

### How Results Are Returned

When the LLM invokes a search function, the system performs these steps internally:

1. The query is preprocessed using the selected NLP backend (synonym expansion, keyword extraction, optional POS tagging).
2. A vector embedding is generated for the query.
3. Vector search retrieves `3x` the requested `count` as candidates.
4. Hybrid scoring combines vector similarity with keyword matching and metadata signals.
5. The top `count` results are selected and formatted.
6. Results are returned to the LLM as the function response, which the LLM uses to formulate its answer to the user.

### Content Length Budgeting

The `max_content_length` parameter controls the total response size. The budget is distributed evenly across results with room for per-result overhead:

```
overhead_per_result = 300 chars  # Metadata, formatting
total_overhead = count * 300
available_for_content = max_content_length - total_overhead
per_result_limit = available_for_content / count
```

For example, with `count=5` and `max_content_length=32768`:
- Each result gets approximately 6,253 characters of content.

This budgeting prevents LLM context exhaustion. If a result exceeds its budget, it is truncated. To detect this, use a format callback (see [Custom Response Formatting](#custom-response-formatting)).

**Tuning content length by use case:**

| Use Case | Recommended max_content_length | Rationale |
|----------|-------------------------------|-----------|
| Voice agents | 16384 (16KB) | Shorter content produces faster, more focused spoken responses |
| Chat agents | 32768 (32KB, default) | Can handle more detailed information in text |
| Deep research queries | 65536 (64KB) | Comprehensive answers requiring lots of context |

### Voice vs Chat Considerations

Voice and chat modes have different requirements for how search results should be presented to the LLM:

- **Voice mode**: Responses should be conversational. URLs, code snippets, and markdown formatting should not be read aloud. Technical concepts should be summarized clearly.
- **Chat mode**: Responses can include URLs, formatted code blocks, links, and comprehensive technical details.

Use the `response_format_callback` parameter (see [Custom Response Formatting](#custom-response-formatting)) to adapt output based on mode. Additionally, prompt the agent to handle each mode appropriately.

---

## Metadata and Tags

Every chunk stored in a search index includes metadata -- structured information beyond the text content itself. Metadata enables filtering, boosting, organization, and provenance tracking.

### Automatic Metadata

The SDK adds metadata automatically based on the chunking strategy used during indexing.

**Sentence strategy:**

```python
{
    "chunk_method": "sentence",
    "chunk_index": 5,
    "sentence_count": 8,
    "filename": "docs.md"
}
```

**Markdown strategy:**

```python
{
    "chunk_method": "markdown",
    "chunk_index": 3,
    "h1": "Getting Started",
    "h2": "Installation",
    "h3": "Python Setup",
    "depth": 3,
    "has_code": True,
    "code_languages": ["python", "bash"],
    "tags": ["code", "code:python", "code:bash"],
    "filename": "installation.md"
}
```

**QA strategy:**

```python
{
    "chunk_method": "qa_optimized",
    "chunk_index": 2,
    "has_question": True,
    "has_process": True,
    "sentence_count": 6
}
```

### Adding Custom Metadata

#### During Indexing (CLI)

Add tags when building the index:

```bash
sw-search ./docs \
  --tags documentation,api,v2,production \
  --output docs.swsearch
```

All chunks in the build receive these tags.

#### In JSON Workflow

When using the JSON chunking strategy, add any metadata fields:

```json
{
  "chunks": [
    {
      "content": "API authentication requires a Bearer token...",
      "metadata": {
        "category": "security",
        "priority": "high",
        "difficulty": "beginner",
        "tags": ["authentication", "security", "api"],
        "last_updated": "2025-01-15",
        "author": "security-team",
        "related_topics": ["authorization", "tokens"],
        "estimated_time": "5 minutes"
      }
    }
  ]
}
```

#### Programmatic Metadata

Generate metadata when creating JSON programmatically:

```python
import json
from datetime import datetime

def create_chunk_with_metadata(content, category, difficulty):
    """Create a chunk with rich metadata"""
    return {
        "content": content,
        "metadata": {
            "category": category,
            "difficulty": difficulty,
            "tags": [category, difficulty, "generated"],
            "created_at": datetime.now().isoformat(),
            "word_count": len(content.split()),
            "has_url": "http" in content,
            "has_code": "```" in content or "def " in content
        }
    }

chunks = {
    "chunks": [
        create_chunk_with_metadata(
            "The AgentBase class provides...",
            category="api-reference",
            difficulty="intermediate"
        ),
        create_chunk_with_metadata(
            "To get started, install the package...",
            category="getting-started",
            difficulty="beginner"
        )
    ]
}

with open("chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)
```

### Tag-Based Filtering

Tags enable precise filtering during search. Specify tags in the skill configuration to restrict which chunks are searched:

```python
# Only search API documentation
self.add_skill("native_vector_search", {
    "tool_name": "search_api",
    "description": "Search API documentation",
    "index_path": "./docs.swsearch",
    "tags": ["api", "reference"]
})
```

Multiple search skills can use the same index with different tag filters:

```python
class DocumentationAgent(AgentBase):
    def __init__(self):
        super().__init__(name="DocsAgent")

        # Beginner-friendly docs
        self.add_skill("native_vector_search", {
            "tool_name": "search_getting_started",
            "description": "Search beginner guides and tutorials",
            "index_path": "./docs.swsearch",
            "tags": ["beginner", "tutorial", "getting-started"]
        })

        # Advanced technical docs
        self.add_skill("native_vector_search", {
            "tool_name": "search_advanced",
            "description": "Search advanced documentation and technical details",
            "index_path": "./docs.swsearch",
            "tags": ["advanced", "technical"]
        })

        # API reference only
        self.add_skill("native_vector_search", {
            "tool_name": "search_api_reference",
            "description": "Search API documentation for classes, methods, and parameters",
            "index_path": "./docs.swsearch",
            "tags": ["api", "reference", "code"]
        })
```

The LLM selects the appropriate search function based on the user's question.

### Metadata Boosting in Hybrid Search

In hybrid search mode, metadata matching provides confirmation signals that boost relevance scores. The boost is applied multiplicatively on top of the vector similarity score.

**Example scenario:** User searches for "python authentication example"

**Chunk A:**
```python
{
    "content": "Here's a Python authentication example...",
    "metadata": {
        "tags": ["python", "authentication", "example", "code"],
        "code_languages": ["python"]
    }
}
```
- Vector similarity: 0.75
- Metadata matches: "python", "authentication", "example" (3 matches) -- boost +30%
- Has "code" tag with keywords matched -- boost +20%
- Final score: 0.75 x 1.30 x 1.20 = 1.17

**Chunk B:**
```python
{
    "content": "Authentication is important for security...",
    "metadata": {
        "tags": ["security", "authentication"]
    }
}
```
- Vector similarity: 0.82 (higher raw score)
- Metadata matches: "authentication" (1 match) -- boost +15%
- Final score: 0.82 x 1.15 = 0.94

**Result:** Chunk A ranks higher despite lower vector similarity because metadata confirmed it matches the user's intent more precisely.

### Organizing by Category, Priority, and Audience

#### Category Organization

Use categories and subcategories to structure large knowledge bases:

```json
{
  "chunks": [
    {
      "content": "...",
      "metadata": {
        "category": "getting-started",
        "subcategory": "installation",
        "tags": ["beginner", "setup"]
      }
    },
    {
      "content": "...",
      "metadata": {
        "category": "api-reference",
        "subcategory": "core-classes",
        "tags": ["api", "reference", "advanced"]
      }
    },
    {
      "content": "...",
      "metadata": {
        "category": "troubleshooting",
        "subcategory": "errors",
        "tags": ["troubleshooting", "debugging", "errors"]
      }
    }
  ]
}
```

Then create category-specific search tools:

```python
self.add_skill("native_vector_search", {
    "tool_name": "search_troubleshooting",
    "description": "Search troubleshooting guides for error solutions",
    "index_path": "./docs.swsearch",
    "tags": ["troubleshooting", "errors"]
})
```

#### Priority Metadata

Mark important content with priority levels:

```json
{
  "content": "Critical security notice: Always validate input...",
  "metadata": {
    "priority": "critical",
    "category": "security",
    "tags": ["security", "important", "critical"]
  }
}
```

Use a custom formatter to surface priority in the response:

```python
def _format_with_priority(self, response, agent, query, results, **kwargs):
    """Highlight high-priority results"""
    formatted = ""

    for result in results:
        priority = result.get('metadata', {}).get('priority', 'normal')

        if priority == 'critical':
            formatted += "CRITICAL: "
        elif priority == 'high':
            formatted += "HIGH PRIORITY: "

        formatted += result['content'] + "\n\n"

    return formatted
```

#### Audience Metadata

Tag content by target audience:

```json
{
  "content": "Advanced memory optimization techniques...",
  "metadata": {
    "audience": "expert",
    "difficulty": "advanced",
    "tags": ["expert", "performance", "optimization"]
  }
}
```

#### Temporal Metadata

Track content freshness:

```json
{
  "content": "New feature in v2.0: async support...",
  "metadata": {
    "version": "2.0",
    "created_at": "2025-01-15",
    "last_updated": "2025-01-20",
    "tags": ["new", "v2", "async"]
  }
}
```

#### Relationship Metadata

Link related chunks for learning paths:

```json
{
  "chunk_id": "auth_overview",
  "content": "Authentication overview...",
  "metadata": {
    "related_chunks": ["auth_examples", "auth_errors"],
    "prerequisite": "installation",
    "next_topic": "authorization"
  }
}
```

#### Language Metadata

For multilingual content, tag by language:

```json
{
  "content": "Le SDK SignalWire permet...",
  "metadata": {
    "language": "fr",
    "translated_from": "en",
    "tags": ["french", "documentation"]
  }
}
```

Filter by language in the skill configuration:

```python
self.add_skill("native_vector_search", {
    "tool_name": "search_french_docs",
    "description": "Rechercher la documentation en francais",
    "index_path": "./docs.swsearch",
    "tags": ["french"]
})
```

### Metadata Best Practices

1. **Be consistent.** Use the same metadata field names across your knowledge base. Avoid mixing `category` and `type`, or `difficulty` and `level`.

2. **Use tags liberally.** More tags create more opportunities for hybrid search boosting:
   ```json
   {
     "tags": ["authentication", "security", "api", "bearer-token", "auth", "login", "credentials"]
   }
   ```

3. **Include synonyms.** Users search with different terms:
   ```json
   {
     "tags": ["installation", "setup", "getting-started", "install", "configure"]
   }
   ```

4. **Structure hierarchically.**
   ```json
   {
     "category": "development",
     "subcategory": "testing",
     "topic": "unit-tests"
   }
   ```

5. **Track provenance.**
   ```json
   {
     "source": "official-docs",
     "author": "signalwire-team",
     "verified": true,
     "last_reviewed": "2025-01-15"
   }
   ```

---

## Tuning Search Quality

### Key Parameters (distance_threshold, count, max_content_length)

Three parameters have the most significant impact on search quality.

#### distance_threshold

Controls how similar results must be to the query. Values range from 0.0 to 1.0, where higher means more similar:

| Score Range | Meaning |
|------------|---------|
| 1.0 | Identical vectors (perfect match) |
| 0.8 | Very similar |
| 0.5 | Somewhat similar |
| 0.2 | Barely related |
| 0.0 | Completely different |

**Recommended thresholds by content type:**

| Content Type | Threshold | Rationale |
|-------------|-----------|-----------|
| Technical documentation (code, APIs) | 0.4 | Specific content; prevents off-topic matches |
| General knowledge base (FAQs, guides) | 0.5 | Conversational content benefits from broader matching |
| Creative content (blogs, articles) | 0.6 | Varied language requires a wider net |
| Precise lookups (error codes, model names) | 0.3 | Exact matches matter; be strict |

**Testing threshold values with the CLI:**

```bash
sw-search search ./docs.swsearch "your query" --threshold 0.3 --verbose
sw-search search ./docs.swsearch "your query" --threshold 0.4 --verbose
sw-search search ./docs.swsearch "your query" --threshold 0.5 --verbose
```

The verbose output displays similarity scores for each result. If the threshold excludes results that appear relevant, lower it. If irrelevant results appear, raise it.

**Dynamic threshold strategy:**

For agents that need reliability across diverse queries, implement fallback logic:

```python
def search_with_fallback(self, query):
    """Search with fallback to lower threshold"""
    # Try strict first
    results = self.search(query, threshold=0.5)

    if len(results) < 2:
        # Not enough results, try more permissive
        results = self.search(query, threshold=0.4)

    if len(results) < 1:
        # Still nothing, try very permissive
        results = self.search(query, threshold=0.3)

    return results
```

#### count

Determines how many results to return. The hybrid search engine internally retrieves `3x` the requested count, scores all candidates, and returns the top results.

**Trade-offs by count range:**

| Range | Advantages | Disadvantages |
|-------|-----------|---------------|
| 1-3 | Precise answers, less noise, faster LLM processing, lower token costs | Might miss context, limited perspective |
| 5-7 | More context, multiple perspectives, better coverage | More noise, slower processing, higher token costs |
| 10+ | Maximum coverage | Information overload, diminishing returns, truncation risk |

**Recommendations by query type:**

- Specific questions ("How do I authenticate?"): `count: 3`
- Exploratory questions ("What are the authentication options?"): `count: 5`
- Research queries ("Tell me everything about authentication"): `count: 7`

#### max_content_length

Controls total response size. See [Content Length Budgeting](#content-length-budgeting) for details on how the budget is distributed.

### Debug Mode

Enable verbose logging to see hybrid scoring details:

```python
import os
os.environ['SEARCH_DEBUG'] = '1'
```

Debug output shows the scoring breakdown for each candidate:

```
Query: "python authentication examples"

Candidate pool: 15 chunks

Chunk 1:
  Vector score: 0.82
  Keyword matches: ["python", "authentication", "examples"]
  Keyword boost: +0.45 (3 matches x 0.15)
  Has 'code' tag: +0.20
  Final score: 0.82 x 1.45 x 1.20 = 1.43

Chunk 2:
  Vector score: 0.75
  Keyword matches: ["authentication"]
  Keyword boost: +0.15 (1 match x 0.15)
  Final score: 0.75 x 1.15 = 0.86

Returning top 5 results...
```

This reveals why certain results rank higher or lower than expected.

Additionally, enable general verbose logging for the agent:

```python
import os
os.environ['SIGNALWIRE_LOG_LEVEL'] = 'DEBUG'
```

### Common Issues and Fixes

#### No Results Returned

**Symptoms:** Search returns empty results for queries that should match content.

**Possible causes and fixes:**
1. **Threshold too strict.** Lower `distance_threshold`: 0.5 to 0.4 to 0.3.
2. **Query phrasing mismatch.** Test variations: "authentication setup", "configuring auth", "setting up authentication".
3. **Content gap.** Verify the topic exists in the index: `sw-search search ./docs.swsearch "test query"`.
4. **Index not loaded.** Check agent logs for errors during skill initialization.

#### Irrelevant Results

**Symptoms:** Results do not match the query intent.

**Possible causes and fixes:**
1. **Threshold too permissive.** Raise `distance_threshold`: 0.4 to 0.5.
2. **Poor metadata tagging.** Add tags for filtering: `tags=["python", "code"]`.
3. **Chunking mixed unrelated content.** Use the markdown chunking strategy for code documentation.

#### Best Result Not Ranked First

**Symptoms:** A clearly relevant result appears below less relevant ones.

**Possible causes and fixes:**
1. **Missing metadata on the best result.** Add relevant tags: `["voice", "configuration"]`.
2. **Hybrid scoring not boosting correctly.** Verify metadata terms match query keywords.
3. **Code chunks missing the "code" tag.** Use the markdown chunking strategy to add automatic code tags.

#### Response Truncated

**Symptoms:** Results appear cut off mid-sentence.

**Possible causes and fixes:**
1. **max_content_length too low.** Increase: 32768 to 65536.
2. **Too many results consuming the budget.** Reduce `count`: 7 to 5.
3. **Individual chunks too long.** Improve chunking to produce shorter chunks.

### A/B Testing Search Configurations

Test different configurations in production to find optimal settings:

```python
import random

class ABTestAgent(AgentBase):
    def __init__(self):
        super().__init__(name="ABTest")

        # Randomly assign configuration
        config_a = {"threshold": 0.4, "count": 5}
        config_b = {"threshold": 0.5, "count": 3}

        config = config_a if random.random() < 0.5 else config_b

        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search documentation",
            "index_path": "./docs.swsearch",
            **config
        })

        # Log which config was used
        logger.info(f"Using config: {config}")
```

**Metrics to track:**
- User satisfaction ratings
- Follow-up question rate (lower is better -- the first answer sufficed)
- Query success rate (non-empty, relevant results)
- Search latency

**Systematic evaluation approach:**

```python
test_queries = [
    "how to create an agent",
    "authentication methods",
    "error handling",
    "voice configuration",
    "deployment options",
    "python code examples",
    "troubleshooting connection issues"
]

configs = [
    {"threshold": 0.3, "count": 3},
    {"threshold": 0.4, "count": 3},
    {"threshold": 0.4, "count": 5},
    {"threshold": 0.5, "count": 5},
]

for config in configs:
    print(f"\nTesting: {config}")
    for query in test_queries:
        results = search(query, **config)
        print(f"  {query}: {len(results)} results")
```

For each configuration, evaluate:
1. **Precision**: Are all returned results relevant?
2. **Recall**: Were all relevant chunks found?
3. **Coverage**: Do results answer the question completely?
4. **Diversity**: Do results cover different aspects of the topic?

---

## Custom Response Formatting

### Format Callback Function

The `response_format_callback` parameter accepts a function that customizes how search results are formatted before being sent to the LLM:

```python
class CustomAgent(AgentBase):
    def __init__(self):
        super().__init__(name="CustomAgent")

        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search documentation",
            "index_path": "./docs.swsearch",
            "response_format_callback": self._format_search_results
        })

    def _format_search_results(self, response, agent, query, results, **kwargs):
        """Custom formatter for search results"""
        if not results:
            return response  # Use default no_results_message

        # Add custom instructions
        formatted = "Documentation Search Results:\n\n"
        formatted += f"Query: {query}\n"
        formatted += f"Found {len(results)} relevant sections:\n\n"

        # Include the default-formatted response
        formatted += response

        # Add footer instructions for the LLM
        formatted += "\n\nBased on these results, provide a clear and accurate answer."

        return formatted
```

**Callback parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | string | The default formatted response |
| `agent` | AgentBase | The agent instance |
| `query` | string | The search query |
| `results` | list[dict] | List of result dictionaries with `content` and `metadata` keys |
| `**kwargs` | dict | Additional metadata (may include `truncated`, `start_time`, etc.) |

### Voice-Optimized Formatting

For agents that handle both voice and chat, use the callback to adapt formatting:

```python
class MultiModalAgent(AgentBase):
    def __init__(self):
        super().__init__(name="MultiModalAgent")

        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search documentation",
            "index_path": "./docs.swsearch",
            "response_format_callback": self._adaptive_format
        })

    def _adaptive_format(self, response, agent, query, results, **kwargs):
        """Format differently for voice vs chat"""
        if not results:
            return response

        is_voice = getattr(agent, 'is_voice', False)

        if is_voice:
            instructions = (
                "Voice Mode:\n"
                "Use these search results to answer naturally. "
                "Do not read URLs or code verbatim. "
                "Summarize technical concepts clearly. "
                "Mention that detailed documentation is available online.\n\n"
            )
        else:
            instructions = (
                "Chat Mode:\n"
                "Use these search results to answer. "
                "Include relevant URLs from results. "
                "Format code with markdown code blocks. "
                "Provide comprehensive technical details.\n\n"
            )

        return instructions + response
```

### Monitoring with Format Callbacks

Use the callback to log search performance metrics:

```python
import time

class MonitoredAgent(AgentBase):
    def __init__(self):
        super().__init__(name="MonitoredAgent")

        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search documentation",
            "index_path": "./docs.swsearch",
            "response_format_callback": self._monitored_format
        })

    def _monitored_format(self, response, agent, query, results, **kwargs):
        """Monitor search performance"""
        start_time = kwargs.get('start_time', time.time())
        search_time = time.time() - start_time

        logger.info(f"Search query: {query}")
        logger.info(f"Results: {len(results)}")
        logger.info(f"Search time: {search_time:.3f}s")

        return response
```

---

## Multiple Search Instances

### Different Collections per Skill

An agent can have multiple `native_vector_search` skill instances, each pointing to a different index or collection. The LLM selects which function to invoke based on the `description` of each tool.

```python
class SupportAgent(AgentBase):
    def __init__(self):
        super().__init__(name="SupportAgent")

        # General documentation
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search general product documentation",
            "index_path": "./docs.swsearch",
            "count": 5
        })

        # API reference
        self.add_skill("native_vector_search", {
            "tool_name": "search_api",
            "description": "Search API documentation for endpoints, parameters, and examples",
            "index_path": "./api.swsearch",
            "tags": ["api"],
            "count": 3
        })

        # Troubleshooting guide
        self.add_skill("native_vector_search", {
            "tool_name": "search_troubleshooting",
            "description": "Search troubleshooting guides for error messages and solutions",
            "index_path": "./troubleshooting.swsearch",
            "tags": ["troubleshooting", "errors"],
            "count": 3
        })
```

### Specialized Search Tools

Combine tag filtering with custom formatters to create specialized search tools from a single index:

```python
class SpecializedAgent(AgentBase):
    def __init__(self):
        super().__init__(name="SpecializedAgent")

        # Code examples only
        self.add_skill("native_vector_search", {
            "tool_name": "find_code_examples",
            "description": "Find code examples and implementation samples",
            "index_path": "./docs.swsearch",
            "tags": ["code", "example"],
            "response_format_callback": self._format_code_examples
        })

        # Error solutions only
        self.add_skill("native_vector_search", {
            "tool_name": "find_error_solutions",
            "description": "Find solutions to error messages and problems",
            "index_path": "./docs.swsearch",
            "tags": ["troubleshooting", "errors", "solutions"]
        })

        # Beginner tutorials only
        self.add_skill("native_vector_search", {
            "tool_name": "find_tutorials",
            "description": "Find beginner-friendly tutorials and guides",
            "index_path": "./docs.swsearch",
            "tags": ["tutorial", "beginner", "guide"]
        })

    def _format_code_examples(self, response, agent, query, results, **kwargs):
        """Format code examples with language annotations"""
        formatted = "Code Examples Found:\n\n"

        for result in results:
            languages = result.get('metadata', {}).get('code_languages', [])
            if languages:
                formatted += f"Languages: {', '.join(languages)}\n"

        formatted += response
        return formatted
```

### Prompt Engineering for Search

When using multiple search tools, instruct the agent on which tool to use for different query types:

```python
class SmartAgent(AgentBase):
    def __init__(self):
        super().__init__(name="SmartAgent")

        self.add_skill("native_vector_search", {
            "tool_name": "search_knowledge",
            "description": "Search our knowledge base",
            "index_path": "./knowledge.swsearch"
        })

        # Instruct agent on search usage
        self.prompt_add_section(
            "Using Search",
            bullets=[
                "ALWAYS search the knowledge base before answering technical questions",
                "Use search_knowledge for questions about features, APIs, or how-to topics",
                "Base your answers on search results, not general knowledge",
                "If search returns no results, tell the user you don't have that information",
                "Don't make up answers - search first, then respond based on results"
            ]
        )
```

---

## Real-World Example

### Multi-Collection Agent (Sigmond Case Study)

Sigmond is SignalWire's production demo agent -- a multi-collection AI assistant that answers questions about SignalWire products, pricing, and FreeSWITCH telephony. It demonstrates how the search system scales from single-agent development to production multi-agent deployment.

### Architecture

Sigmond uses three separate knowledge bases stored in a single PostgreSQL database with the pgvector extension:

```
PostgreSQL (pgvector)
+-- signalwire_unified (5,000+ chunks)
|   +-- SDK documentation
|   +-- Developer guides
|   +-- API references
|   +-- Platform features
+-- pricing (500+ chunks)
|   +-- Pricing pages
|   +-- Plan comparisons
|   +-- Billing information
+-- freeswitch (2,000+ chunks)
    +-- FreeSWITCH documentation
    +-- Telephony concepts
    +-- SIP configuration
```

Each collection uses a chunking strategy optimized for its content type:

**SignalWire unified documentation** -- built with the markdown strategy to preserve code blocks and header hierarchy:

```bash
sw-search \
  ./signalwire-docs \
  ./sdk-docs \
  ./api-docs \
  --chunking-strategy markdown \
  --model mini \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name signalwire_unified \
  --tags documentation,signalwire,api,sdk
```

**Pricing collection** -- built with the JSON strategy for precise control over chunk boundaries (each plan, feature, or price point as its own chunk):

```bash
sw-search \
  ./pricing.json \
  --chunking-strategy json \
  --model mini \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name pricing \
  --tags pricing,plans,costs
```

**FreeSWITCH documentation** -- built with markdown strategy for technical content:

```bash
sw-search \
  ./freeswitch-docs \
  --chunking-strategy markdown \
  --model mini \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name freeswitch \
  --tags freeswitch,telephony,sip
```

**Agent configuration with all three search skills:**

```python
class SigmondAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Sigmond",
            route="/sigmond",
            port=3000
        )

        # Build pgvector connection
        pg_user = os.getenv("PGVECTOR_DB_USER", "signalwire")
        pg_pass = os.getenv("PGVECTOR_DB_PASSWORD")
        pg_host = os.getenv("PGVECTOR_HOST", "localhost")
        pg_port = os.getenv("PGVECTOR_PORT", "5432")
        pg_db = os.getenv("PGVECTOR_DB_NAME", "knowledge")

        connection_string = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        self._add_search_skills(connection_string)

    def _add_search_skills(self, connection_string):
        """Add three specialized search skills"""

        # 1. SignalWire unified documentation
        self.add_skill("native_vector_search", {
            "tool_name": "search_signalwire_knowledge",
            "description": "Search all SignalWire knowledge including SDK documentation, developer docs, API references, and general platform information",
            "backend": "pgvector",
            "connection_string": connection_string,
            "collection_name": "signalwire_unified",
            "model_name": "mini",
            "count": 5,
            "distance_threshold": 0.4,
            "response_format_callback": self._format_search_results,
            "no_results_message": "I couldn't find information about '{query}' in the SignalWire knowledge base.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search the SignalWire knowledge base...",
                        "I'm looking through the documentation...",
                        "Searching for SignalWire information...",
                        "Let me check the technical documentation..."
                    ]
                }
            }
        })

        # 2. Pricing information
        self.add_skill("native_vector_search", {
            "tool_name": "search_pricing",
            "description": "Search for SignalWire pricing information, plans, costs, and billing details",
            "backend": "pgvector",
            "connection_string": connection_string,
            "collection_name": "pricing",
            "model_name": "mini",
            "count": 3,
            "distance_threshold": 0.4,
            "response_format_callback": self._format_search_results,
            "no_results_message": "I couldn't find specific pricing information for '{query}'. Please check signalwire.com/pricing or contact sales@signalwire.com.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me check the pricing information...",
                        "Looking up pricing details...",
                        "Searching pricing data..."
                    ]
                }
            }
        })

        # 3. FreeSWITCH documentation
        self.add_skill("native_vector_search", {
            "tool_name": "search_freeswitch_knowledge",
            "description": "Search for knowledge about FreeSWITCH telephony system",
            "backend": "pgvector",
            "connection_string": connection_string,
            "collection_name": "freeswitch",
            "model_name": "mini",
            "count": 3,
            "distance_threshold": 0.4,
            "response_format_callback": self._format_search_results,
            "no_results_message": "I couldn't find information about '{query}' in the FreeSWITCH documentation.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search the FreeSWITCH documentation...",
                        "Looking through FreeSWITCH knowledge...",
                        "Searching FreeSWITCH information..."
                    ]
                }
            }
        })

    def _format_search_results(self, response, agent, query, results, **kwargs):
        """Custom formatter that adapts to voice vs chat mode"""
        if not results:
            return response  # Use default no_results_message

        is_voice = getattr(agent, 'is_voice', False)

        if is_voice:
            instructions = (
                "Voice Mode Instructions:\n"
                "Use the following search results to answer the user's question. "
                "Since this is a voice conversation:\n"
                "- Provide a natural, conversational response\n"
                "- Do not read URLs or code snippets verbatim\n"
                "- Summarize technical concepts clearly\n"
                "- Mention that code examples and links are available in the developer docs\n"
                "- Keep responses concise and easy to follow by ear\n"
                "- If there is not enough info in the response, try searching the web\n\n"
            )
        else:
            instructions = (
                "Chat Mode Instructions:\n"
                "Use the following search results to answer the user's question. "
                "Since this is a text chat:\n"
                "- Include relevant URLs from the results in your response\n"
                "- Format all code examples with markdown code blocks\n"
                "- Scrape any relevant URLs to get more detailed information\n"
                "- Provide comprehensive technical details when appropriate\n"
                "- Use markdown formatting for better readability\n"
                "- If there is not enough info in the response, try searching the web\n\n"
            )

        return instructions + response
```

### Prompt Engineering for Search

Sigmond's prompt explicitly maps query types to search tools:

```python
self.prompt_add_section(
    "Using Your Tools",
    body="Match the right tool to each question:",
    bullets=[
        "SignalWire technical/SDK/API questions -> search_signalwire_knowledge",
        "Pricing/costs questions -> search_pricing",
        "FreeSWITCH/telephony questions -> search_freeswitch_knowledge",
        "Current events/general info -> web_search",
        "Specific URLs -> scrape_url or crawl_site",
        "ALWAYS search before answering technical questions."
    ]
)

self.prompt_add_section(
    "Your Mission",
    bullets=[
        "For SignalWire questions: ALWAYS search_signalwire_knowledge first, then answer.",
        "For pricing: ALWAYS search_pricing first. Mention transparent developer pricing and sales@signalwire.com.",
        "You showcase the AI Kernel - fast, native infrastructure without latency."
    ]
)
```

This explicit mapping increased search tool usage from 60% to 95% of queries.

### Performance Metrics and Lessons Learned

**Query performance (production):**

| Collection | Chunk Count | Avg Query Time |
|-----------|-------------|---------------|
| signalwire_unified | 5,000+ | ~30ms |
| pricing | 500+ | ~10ms |
| freeswitch | 2,000+ | ~20ms |

**Success rates:**
- 95% of technical questions answered from search
- 98% of pricing questions answered from search
- 5% fallback to web search for current information

**Deployment configuration:**

All pods share the same pgvector database, eliminating index duplication:

```
Kubernetes Deployment
+-- sigmond-pod-1 --+
+-- sigmond-pod-2 --+--> PostgreSQL (pgvector)
+-- sigmond-pod-3 --+
+-- sigmond-pod-4 --+
```

Production Dockerfile using the lightweight `search-queryonly` installation (pods do not need ML models -- they query pre-built indexes):

```dockerfile
FROM python:3.11-slim
RUN pip install signalwire-agents[search-queryonly]
COPY sigmond.py /app/
CMD ["python", "/app/sigmond.py"]
```

Environment variables:

```bash
export PGVECTOR_DB_USER=signalwire
export PGVECTOR_DB_PASSWORD=<secure-password>
export PGVECTOR_DB_NAME=sigmond_knowledge
export PGVECTOR_HOST=postgres.production.local
export PGVECTOR_PORT=5432
```

**Lessons learned:**

1. **Multiple collections outperform a single index.** A single large index mixed SDK docs with pricing and FreeSWITCH content, producing confused results. Separate collections allow the LLM to choose the right search and return focused results.

2. **Markdown strategy is essential for technical documentation.** Sentence chunking split code blocks mid-code and lost header context. The markdown strategy keeps code blocks intact, preserves header hierarchy, and adds automatic "code" tags that boost code example results.

3. **Voice and chat require different formatting.** Without adaptive formatting, voice mode read URLs aloud and chat mode lacked links and code blocks. The `response_format_callback` solves this cleanly.

4. **Explicit prompt instructions drive search usage.** Vague instructions like "You have search functions" resulted in only 60% search usage. Explicit tool-to-query-type mapping raised it to 95%.

5. **The mini embedding model is sufficient for most use cases.** Testing mini vs base showed approximately 2% quality difference but 2x speed improvement and 50% smaller index size.

6. **A distance_threshold of 0.4 is the production sweet spot for technical documentation.** Testing across 0.3 to 0.6 showed 0.3 was too strict (frequent zero results), 0.5 included some irrelevant results, and 0.6 was too permissive.

7. **Rolling updates enable zero-downtime collection changes.** Build the new collection under a new name, test it, switch agents via environment variable, then delete the old collection.

**Cost analysis (pgvector):**
- Database hosting: ~$100/month for 20GB
- Supports 10+ concurrent agents, 1,000+ queries/day
- More cost-effective than managed vector database alternatives at this scale

**Scalability path:**
- Add read replicas for increased throughput
- Partition collections for millions of chunks
- Connection pooling handles 100+ agents

For more information on deployment patterns, see [search_deployment.md](search_deployment.md). For troubleshooting production issues, see [search_troubleshooting.md](search_troubleshooting.md).

---

## Testing Search Integration

Use `swaig-test` to verify search works before deployment:

```bash
# List available tools
swaig-test agent.py --list-tools

# Test search function
swaig-test agent.py --exec search_docs --query "how to create an agent"
```

Verify the index directly:

```bash
sw-search search ./knowledge.swsearch "test query"
```

For comprehensive testing approaches including A/B testing, see the [Tuning Search Quality](#tuning-search-quality) section.

## Examples

- `examples/search_with_custom_formatter.py` - Custom response formatter callback for search results
- `examples/sigmond_simple.py` - Simple agent with local `.swsearch` file-based knowledge search
- `examples/sigmond_native_search.py` - Native vector search skill with local search index
- `examples/pgvector_search_agent.py` - PGVector backend for document search
