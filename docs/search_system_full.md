# 1. Introduction: Why Built-in Search Matters for AI Agents

If you've built AI agents, you've encountered the hallucination problem. Ask your agent about your API documentation, and it confidently invents endpoints that don't exist. Ask about your product's features, and it describes capabilities you haven't built yet. The model is helpful, creative, and completely wrong.

## The Hallucination Problem

Large language models are trained on vast amounts of internet text, but they don't know about your specific documentation, your company's internal knowledge base, or the details of your product. When asked questions outside their training data, they don't say "I don't know" - they generate plausible-sounding answers that are often completely fictional.

For AI agents representing your business, this is unacceptable. Your agent needs to answer questions accurately based on your actual documentation, not make things up.

## RAG: Retrieval-Augmented Generation

The solution is called RAG - Retrieval-Augmented Generation. The concept is simple:

1. When a user asks a question, search your knowledge base for relevant information
2. Include that information in the prompt to the LLM
3. Instruct the LLM to answer based on the retrieved information

Instead of relying on the model's training data, you're giving it access to your actual documentation at query time. The model becomes a natural language interface to your knowledge base, not a creative fiction writer.

## Why We Built Search Into the SDK

Most RAG implementations require you to:
- Set up a separate vector database (Pinecone, Weaviate, Qdrant, etc.)
- Write code to manage embeddings
- Handle the retrieval logic yourself
- Pay for additional infrastructure
- Deal with latency from multiple service calls

We built search directly into the SignalWire Agents SDK because:

**It should be simple**: `pip install signalwire-agents[search]` and you have everything you need. No separate services to configure.

**It should be portable**: A `.swsearch` file contains your entire knowledge base - embeddings and metadata in a single file you can deploy anywhere.

**It should be integrated**: The `native_vector_search` skill works like any other agent skill. No custom code required.

**It should be flexible**: Start with local SQLite files for development, scale to PostgreSQL pgvector for production. Same API, different backends.

## What Makes Our Approach Different

### Self-Contained Knowledge

Your `.swsearch` files are complete, portable knowledge bases. Build them once, deploy them with your agent. No external dependencies at query time (with the `search-queryonly` install option).

### Hybrid Search

We don't just do vector similarity. Our search combines:
- **Vector embeddings** for semantic understanding
- **Keyword matching** for exact terms
- **Metadata filtering** for structured queries
- **Intelligent scoring** that boosts results matching multiple signals

### Production-Ready Backends

Use SQLite `.swsearch` files for simple deployments, or PostgreSQL pgvector for multi-agent production systems. The same code works with both backends.

### Built for Voice and Chat

Special response formatting adapts search results for voice conversations (don't read URLs aloud) vs text chat (include links and code examples).

### Multiple Chunking Strategies

Your documentation isn't all the same. We provide chunking strategies optimized for:
- Technical documentation with code examples (markdown strategy)
- PDFs and long-form content (page strategy)
- Q&A and tutorial content (qa strategy)
- Pre-structured content you've manually curated (json strategy)

## What's Next

In this guide, we'll take you from zero to building production agents with powerful search capabilities. You'll learn:
- How to create your first search index
- How vector search actually works
- How to choose the right chunking strategy
- How to deploy at scale with pgvector
- How real-world agents like Sigmond use search

Let's get started.
# 2. Getting Started: Your First Search Index

Let's build a simple AI agent that can answer questions about your documentation. We'll start with the basics and have a working example in just a few minutes.

## Installing Search Dependencies

The SignalWire Agents SDK has modular dependencies to keep installations lean. For search functionality, install with the `search` extra:

```bash
pip install signalwire-agents[search]
```

This installs the core search functionality with sentence-transformers for embeddings (~500MB total). If you need advanced document processing (PDFs, Word docs, etc.), use:

```bash
pip install signalwire-agents[search-full]
```

For production deployments where you only need to query existing indexes (not build them), use:

```bash
pip install signalwire-agents[search-queryonly]
```

This is significantly smaller (~400MB) because it doesn't include the ML models needed to build indexes.

## Creating Your First .swsearch File

Let's create a simple knowledge base from a directory of markdown files. Say you have a `docs/` folder with your documentation:

```bash
sw-search ./docs --output knowledge.swsearch
```

That's it. The `sw-search` command will:
1. Recursively scan the `docs/` directory
2. Extract text from supported file types (markdown, txt, etc.)
3. Break the content into chunks
4. Generate vector embeddings for each chunk
5. Save everything to `knowledge.swsearch`

You'll see output like:

```
Processing files...
✓ docs/getting-started.md (3 chunks)
✓ docs/api-reference.md (12 chunks)
✓ docs/examples.md (5 chunks)

Building index...
✓ Generated embeddings for 20 chunks
✓ Index saved to knowledge.swsearch

Index size: 2.3 MB
Total chunks: 20
```

## Querying the Index from an Agent

Now let's create an agent that can search this knowledge base:

```python
#!/usr/bin/env python3
"""
Simple agent with search capabilities
"""
import os
from signalwire_agents import AgentBase

class DocsAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="DocsAgent",
            route="/docs",
            port=3000
        )

        # Add the search skill
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the documentation for information",
            "remote_url": "http://localhost:8001",
            "index_name": "knowledge",
            "count": 3,
            "distance_threshold": 0.4
        })

        # Build the prompt
        self.prompt_add_section(
            "Your Role",
            bullets=[
                "You are a helpful documentation assistant.",
                "When users ask questions, use the search_docs function to find relevant information.",
                "Always search before answering technical questions.",
                "Provide accurate answers based on the search results."
            ]
        )

        self.add_language(
            name="English",
            code="en-US",
            voice="elevenlabs.adam"
        )

if __name__ == "__main__":
    agent = DocsAgent()
    print(f"Agent running at: {agent.get_full_url()}")
    agent.run()
```

## Using a Local .swsearch File Directly

Instead of running a search server, you can point directly to your `.swsearch` file:

```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search the documentation for information",
    "index_path": "./knowledge.swsearch",  # Direct path to .swsearch file
    "count": 3,
    "distance_threshold": 0.4
})
```

This is perfect for simple deployments. The agent loads the index at startup and queries it directly.

## Complete Working Example

Here's a complete example with better configuration:

```python
#!/usr/bin/env python3
import os
from signalwire_agents import AgentBase

class DocsAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="DocsAgent",
            route="/docs",
            port=3000
        )

        # Configure search skill
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the documentation to answer user questions about features, APIs, and how-to guides",
            "index_path": "./knowledge.swsearch",
            "count": 5,  # Return top 5 results
            "distance_threshold": 0.4,  # Only results with similarity > 0.4
            "no_results_message": "I couldn't find information about '{query}' in the documentation. Could you rephrase your question?",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search the documentation for that...",
                        "I'm looking through the docs...",
                        "Searching for information..."
                    ]
                }
            }
        })

        # Build agent personality
        self.prompt_add_section(
            "Your Role",
            bullets=[
                "You are a friendly and knowledgeable documentation assistant.",
                "Your goal is to help users find information quickly and accurately."
            ]
        )

        self.prompt_add_section(
            "How to Answer Questions",
            bullets=[
                "Always use search_docs to find information before answering.",
                "Base your answers on the search results, not general knowledge.",
                "If the search returns no results, admit you don't have that information.",
                "Keep answers concise but complete.",
                "When appropriate, ask clarifying questions."
            ]
        )

        # Configure voice
        self.add_language(
            name="English",
            code="en-US",
            voice="elevenlabs.adam",
            function_fillers=["One moment...", "Let me check that..."]
        )

if __name__ == "__main__":
    agent = DocsAgent()
    print("=" * 60)
    print(f"Documentation Agent")
    print(f"Available at: {agent.get_full_url()}")
    print("=" * 60)
    agent.run()
```

## Testing Your Agent

Run the agent:

```bash
python docs_agent.py
```

Then test it with the `swaig-test` CLI:

```bash
swaig-test docs_agent.py --list-tools
# Shows: search_docs

swaig-test docs_agent.py --exec search_docs --query "how do I create an agent"
```

You'll see the search results returned as a formatted response with the most relevant documentation chunks.

## What Just Happened?

When you run a search:

1. Your query is converted to a vector embedding
2. The index compares your query vector to all chunk vectors
3. Chunks are scored by similarity (cosine distance)
4. Keyword and metadata matches boost scores
5. Top results (with distance < threshold) are returned
6. Results are formatted and returned to the LLM
7. The LLM uses the results to answer the user's question

## Next Steps

Now that you have a basic working agent with search, let's dive deeper into how vector search actually works and why it's so much better than keyword search for natural language queries.
# 3. Understanding Vector Search

Vector search sounds complicated, but the core concept is elegant: convert text into numbers that capture meaning, then use math to find similar meanings.

## What Are Embeddings? (Simple Explanation)

An embedding is a list of numbers that represents the meaning of a piece of text. Think of it like GPS coordinates for ideas.

Just as GPS coordinates let you calculate which locations are close together, embeddings let you calculate which pieces of text have similar meanings. The math is different (cosine distance vs geographic distance), but the concept is the same: proximity equals similarity.

Here's a simplified example (real embeddings have hundreds of dimensions):

```
"How do I make a phone call?"     → [0.2, 0.8, 0.1, ...]
"Initiating voice connections"    → [0.3, 0.7, 0.2, ...]
"Pizza recipes"                    → [0.9, 0.1, 0.8, ...]
```

The first two are close together in vector space (similar meanings), while the third is far away (different topic).

## Why Vector Search Beats Keyword Search

Traditional keyword search looks for exact word matches. If your documentation says "initiating voice connections" and the user asks "how do I make a phone call", keyword search finds nothing - not a single word matches.

Vector search finds it immediately because the meaning is similar.

### Real-World Example

Let's say you have this in your documentation:

> "The AgentBase class provides methods for configuring voice parameters, including setting the voice model and adjusting speech characteristics."

**User questions that keyword search would miss:**

- "How do I change the AI's voice?" (no word "change" or "AI" in docs)
- "Can I configure speech settings?" (different phrasing)
- "Where do I set voice options?" (synonyms)

**Vector search finds them all** because it understands:
- "change" ≈ "configuring" ≈ "setting"
- "AI's voice" ≈ "voice model"
- "speech settings" ≈ "speech characteristics"

## Semantic Similarity in Action

The magic of embeddings is that similar concepts cluster together in vector space, even with completely different words:

```
Cluster 1 (Authentication concepts):
- "logging in"
- "user credentials"
- "authentication tokens"
- "sign in process"

Cluster 2 (Error handling):
- "dealing with failures"
- "error management"
- "handling exceptions"
- "when things go wrong"

Cluster 3 (Getting started):
- "initial setup"
- "first time configuration"
- "installation steps"
- "beginning your project"
```

A query like "how do I authenticate users" will find documents in Cluster 1, even if they never use the word "authenticate".

## How Embedding Models Work

The embedding models we use (like `sentence-transformers/all-MiniLM-L6-v2`) are neural networks trained on massive amounts of text to learn these semantic relationships.

They've learned that:
- "car" and "automobile" are synonyms
- "doctor" is related to "hospital" and "medicine"
- "happy" is opposite to "sad"
- "Python" can mean a snake or a programming language (context matters)

When you run `sw-search` to build an index, the model:
1. Reads each chunk of text
2. Processes it through the neural network
3. Outputs a vector (typically 384 or 768 numbers)
4. Stores that vector alongside the original text

At query time:
1. Your search query goes through the same model
2. Generates a query vector
3. Compares it to all chunk vectors
4. Returns the closest matches

## The Math: Cosine Similarity

The comparison uses cosine similarity, which measures the angle between two vectors. Vectors pointing in the same direction (similar meanings) have high similarity (close to 1.0), while vectors pointing in different directions have low similarity (close to 0.0).

You don't need to understand the math, but here's the intuition:

```
Query: "how to handle errors"
Chunk A: "error handling guide"     → similarity: 0.87 (very similar)
Chunk B: "installation instructions" → similarity: 0.23 (not similar)
```

The `distance_threshold` parameter (e.g., 0.4) filters out low-similarity results. Only chunks with similarity above the threshold are returned.

## When Keyword Search Still Matters

Vector search is powerful, but it's not perfect. Sometimes you need exact matches:

**Exact terms matter:**

- Model names: "gpt-4o-mini" vs "GPT-4"
- API endpoints: "/api/v2/users" vs "/api/users"
- Error codes: "ERROR_404" vs "ERROR_500"
- Programming identifiers: "set_params()" vs "get_params()"

**Solution: Hybrid Search**

That's why our search system combines vector similarity with keyword matching. If someone searches for "gpt-4o-mini", we boost results that contain that exact string, even if the vector similarity isn't the highest.

This is the best of both worlds:
- Vector search finds semantically similar content
- Keyword matching ensures exact terms aren't missed
- Metadata filtering adds structured querying
- Combined scoring ranks the best results highest

## Embeddings Are Fast

Once embeddings are generated, search is extremely fast. Comparing vectors is just arithmetic - multiply and add operations. Modern CPUs can compare thousands of vectors per millisecond.

That's why `.swsearch` files work so well: the expensive part (generating embeddings) happens once during indexing. Queries are cheap.

## Understanding Distance Threshold

The `distance_threshold` parameter controls how strict matching is:

```python
"distance_threshold": 0.3  # Very strict - only near-perfect matches
"distance_threshold": 0.5  # Balanced - good matches
"distance_threshold": 0.7  # Permissive - includes loosely related content
```

For technical documentation, 0.4-0.5 works well. For creative content or broad topics, 0.6-0.7 might be better.

Too strict? You get no results.
Too permissive? You get irrelevant results.

Start with 0.4 and adjust based on your content.

## What Makes a Good Embedding Model?

Different models have different characteristics:

**MiniLM (our default "mini"):**

- 384 dimensions
- Very fast (~5x faster than base)
- Great for most use cases
- Lower memory usage

**MPNet (our "base" model):**

- 768 dimensions
- Better quality for complex queries
- Slower but more accurate
- Previous default

The quality difference is often negligible for well-written documentation. The speed difference matters a lot for large indexes.

## Key Takeaways

1. **Embeddings capture meaning** as vectors of numbers
2. **Similar meanings = similar vectors** even with different words
3. **Vector search finds semantic similarity** that keyword search misses
4. **Hybrid search combines both approaches** for best results
5. **Embeddings are generated once**, queries are fast
6. **distance_threshold controls strictness** of matching

Now that you understand how vector search works, let's explore how our hybrid search algorithm combines vector similarity with keyword matching to deliver even better results.
# 4. Hybrid Search: Best of Both Worlds

Vector search is powerful for semantic similarity, but it can miss important exact matches. Keyword search is great for finding specific terms, but it fails when users ask questions differently. Hybrid search combines both approaches to deliver superior results.

## Vector-First Architecture

Our hybrid search is **vector-first**, not keyword-first. Here's why that matters.

### The Old Way (Keyword-First)

Many RAG systems work like this:

1. Try keyword search first
2. If that fails, fall back to vector search
3. Return whatever you found

This treats vector search like a backup plan. It works like grep with a safety net.

**Problem:** You miss semantically relevant content that doesn't have keyword matches. You're still doing keyword search, just with a backup.

### Our Way (Vector-First)

Our approach is fundamentally different:

1. **Always run vector search** as the primary signal
2. **Run keyword/metadata searches in parallel** (not conditionally)
3. **Use keyword matches as confirmation signals** that boost scores
4. **Return the best combined results**

Vector search drives the results. Keyword and metadata matches boost the scores of results that match on multiple signals.

## How Hybrid Scoring Works

Let's walk through a concrete example.

**Query:** "how to configure voice settings"

**Step 1: Vector Search (Primary Signal)**

Search for 3x the requested number of results (if you want 5 results, search for 15). This gives us a candidate pool:

```
Candidate Results (by vector similarity):
1. "The set_params() method configures AI parameters..." (0.82)
2. "Voice configuration uses the add_language() method..." (0.78)
3. "Adjusting speech characteristics for your agent..." (0.75)
4. "Setting up video parameters for video calls..." (0.71)
5. "Database connection configuration options..." (0.45)
...
```

**Step 2: Keyword/Metadata Search (In Parallel)**

While vector search runs, we also search for:
- **Keywords:** "configure", "voice", "settings"
- **Metadata:** chunks tagged with "configuration", "voice", "audio"
- **Filenames:** files named like "voice_config.md"

**Step 3: Combined Scoring**

Now we boost results that match on multiple signals:

```
Result 2: Vector (0.78) + Keyword match ("voice", "configuration") + Metadata tag (voice)
→ Base score: 0.78
→ Keyword boost: +15% (matched 2 keywords)
→ Metadata boost: +15% (has "voice" tag)
→ Final score: 0.78 × 1.30 = 1.01

Result 1: Vector (0.82) + No keyword matches
→ Final score: 0.82 (no boost)

Result 2 now ranks HIGHER than Result 1
```

Results matching on both semantic similarity AND exact terms get boosted. This is the confirmation signal - when multiple independent signals agree, we have higher confidence.

**Step 4: Special Boosts**

We apply special boosts for high-value content:

**Code chunks get 20% boost** (if keyword/metadata also matched):
- Chunks with code blocks are more valuable for technical queries
- If a chunk has a "code" tag AND matches keywords, boost by 20%
- This helps surface code examples when users ask "how do I" questions

```
Result with code: 0.70 (vector) × 1.15 (keyword) × 1.20 (code tag) = 0.97
Result without code: 0.85 (vector) × 1.0 (no boost) = 0.85
→ Code example ranks higher!
```

## Why This Works Better

### Example: Finding Code Examples

**Documentation chunk:**

```markdown
## Configuring Voice Parameters

You can configure voice settings using the add_language method:


agent.add_language(
    name="English",
    code="en-US",
    voice="elevenlabs.adam"
)

```

**User query:** "show me python code for setting up voice"

**Vector-only search:** Might return conceptual documentation that's semantically similar but has no code.

**Hybrid search:**

- Vector similarity finds voice configuration topics (0.75)
- Keywords "python" and "voice" match (+15%)
- Chunk has "code" tag (+20% because keywords matched)
- **Final score boosted from 0.75 to 1.03**

This chunk rockets to the top because it matches on all signals: semantics, keywords, and content type.

## The Confirmation Principle

The core insight of hybrid search is **confirmation**:

When multiple independent signals agree that a chunk is relevant, you can be more confident it's actually relevant.

Think of it like a jury:
- Vector search is one juror: "This seems related"
- Keyword search is another: "It has the exact terms"
- Metadata is a third: "It's tagged as relevant"

When all three agree, you have a strong verdict.

## Metadata as a Signal

Metadata adds structure to semantic search:

```python
# Chunk metadata
{
    "tags": ["api", "reference", "authentication"],
    "section": "API Reference",
    "h1": "Authentication",
    "has_code": True,
    "code_languages": ["python", "javascript"]
}
```

**Query:** "python authentication example"

**Metadata boosts:**

- Has "authentication" tag → +boost
- Has "code" tag → +boost
- Has "python" in code_languages → +boost
- In "API Reference" section → +boost

Combined with vector similarity, this chunk gets a strong score.

## Real-World Performance

We redesigned our search algorithm after discovering that results were too keyword-focused and missing relevant code examples.

**Before (keyword-first):**

- Query: "example code Python SignalWire Agents SDK"
- Results: Conceptual docs about examples, no actual code
- Problem: Treated vector search like grep

**After (vector-first with hybrid scoring):**

- Same query
- Results: Actual Python code examples from SDK documentation
- Reason: Code chunks with matching metadata got boosted

The difference is dramatic for technical documentation where users need code examples.

## Tuning Hybrid Search

You can influence hybrid behavior through configuration:

### Distance Threshold

```python
"distance_threshold": 0.4  # Only vector results > 0.4 enter the pool
```

Stricter thresholds mean fewer candidates for keyword boosting. Looser thresholds let more results benefit from keyword/metadata boosts.

### Result Count

```python
"count": 5  # Returns top 5 after hybrid scoring
```

Internally, we search for 3x this number (15) to have a good candidate pool. After hybrid scoring, we return the top 5.

### Custom Metadata

Add your own metadata during indexing:

```python
chunk_metadata = {
    "tags": ["tutorial", "beginner", "quickstart"],
    "difficulty": "easy",
    "category": "getting-started"
}
```

Queries matching this metadata will get boosted in hybrid scoring.

## The Algorithm in Detail

For those who want the technical details:

```
1. Vector search for N*3 candidates (where N = requested count)
   → Generates pool of semantically similar chunks

2. Parallel keyword search for query terms
   → Identifies chunks with exact term matches

3. Parallel metadata search for tags/fields
   → Identifies chunks with matching metadata

4. For each candidate chunk:
   base_score = vector_similarity

   if chunk matched keywords:
       keyword_boost = min(0.30, num_keywords * 0.15)
       base_score *= (1.0 + keyword_boost)

   if chunk matched metadata:
       metadata_boost = min(0.30, num_metadata_matches * 0.15)
       base_score *= (1.0 + metadata_boost)

   if chunk has 'code' tag AND (keywords matched OR metadata matched):
       base_score *= 1.20

   final_score = base_score

5. Sort by final_score descending

6. Return top N results
```

The boost percentages are tuned based on real-world testing with technical documentation.

## When Hybrid Search Shines

Hybrid search excels when:

- **Users ask questions with specific terminology** - both semantic and exact matching matter
- **Content has structure** - metadata adds valuable signals
- **Code examples matter** - special boosting surfaces code chunks
- **Synonyms and exact terms both appear** - captures both variants
- **Multiple related topics exist** - confirmation signals disambiguate

## Key Takeaways

1. **Vector-first, not keyword-first** - semantics drive results
2. **Keyword/metadata as confirmation** - boost results matching multiple signals
3. **Code chunks get special treatment** - 20% boost for code when relevant
4. **Multiple signals = higher confidence** - the confirmation principle
5. **Tunable via configuration** - adjust thresholds and counts for your content

Next, we'll explore how chunking strategy affects what content makes it into your search index and how it impacts search quality.
# 5. Chunking Strategies: Breaking Documents Apart

Before your content can be searched, it must be broken into chunks. This seemingly simple step has enormous impact on search quality. Choose the wrong chunking strategy, and your agent won't find the information users need.

## Why Chunking Matters

LLMs have context limits. You can't send an entire 100-page manual in a single prompt. Even if you could, the LLM would struggle to extract the relevant information from so much text.

Chunking solves this by:
- Breaking documents into search-friendly pieces
- Keeping related information together
- Creating enough context for meaningful answers
- Fitting within embedding model limits (typically 512 tokens)

**The goal:** Each chunk should be a self-contained unit that answers a question or explains a concept.

## The Trade-offs

Chunking is always a trade-off:

**Small chunks:**

- ✅ Precise matching
- ✅ Less noise in results
- ❌ May lack context
- ❌ Can split related information

**Large chunks:**

- ✅ More context
- ✅ Keep related info together
- ❌ Less precise matching
- ❌ More noise in results

Your content determines the best strategy.

## Built-in Strategies Overview

The SDK provides nine chunking strategies, each optimized for different content types.

### 1. Sentence-Based (Default)

**How it works:** Groups sentences together, splitting at natural sentence boundaries.

```bash
sw-search ./docs --chunking-strategy sentence --max-sentences-per-chunk 5
```

**Parameters:**

- `max-sentences-per-chunk`: Number of sentences per chunk (default: 5)
- `split-newlines`: Minimum newlines to force a split (default: 2)

**Best for:**

- General documentation
- Blog posts
- Plain text content
- When you're not sure what to use

**Example:**

```
Original text:
"The AgentBase class is the foundation of all agents. It provides methods for configuration. You can customize voice, prompts, and behavior. Here's a simple example. Create an instance and call run()."

Chunks (5 sentences each):
Chunk 1: "The AgentBase class is the foundation of all agents. It provides methods for configuration. You can customize voice, prompts, and behavior. Here's a simple example. Create an instance and call run()."
```

### 2. Paragraph

**How it works:** Splits at paragraph boundaries (double newlines).

```bash
sw-search ./docs --chunking-strategy paragraph
```

**Best for:**

- Content with clear paragraph structure
- Markdown files with distinct sections
- When paragraphs are already well-sized

**When to avoid:** Dense text with very long paragraphs (chunks become too large).

### 3. Page

**How it works:** Splits documents at page boundaries.

```bash
sw-search ./docs --chunking-strategy page --file-types pdf
```

**Best for:**

- PDF documents
- Content where page breaks are meaningful
- Presentations
- Reports with page-level organization

**When to avoid:** Web content, markdown files (no concept of pages).

### 4. Sliding Window

**How it works:** Creates overlapping chunks by sliding a window across the text.

```bash
sw-search ./docs \
  --chunking-strategy sliding \
  --chunk-size 100 \
  --overlap-size 20
```

**Parameters:**

- `chunk-size`: Words per chunk
- `overlap-size`: Words that overlap between chunks

**Best for:**

- Ensuring context isn't lost at chunk boundaries
- Dense technical content where breaks are arbitrary
- When you want redundancy

**Trade-off:** More chunks (larger index) due to overlap, but better recall.

**Example:**

```
Text: "A B C D E F G H I J"
Chunk size: 5, Overlap: 2

Chunk 1: A B C D E
Chunk 2:     D E F G H
Chunk 3:         G H I J
```

### 5. Semantic

**How it works:** Groups sentences with similar embeddings together.

```bash
sw-search ./docs \
  --chunking-strategy semantic \
  --semantic-threshold 0.6
```

**Parameters:**

- `semantic-threshold`: Similarity threshold for grouping (0.0-1.0)

**Best for:**

- Content that naturally clusters by topic
- Long documents with shifting topics
- When logical boundaries are unclear

**When to avoid:** Short documents, already well-structured content (overhead not worth it).

### 6. Topic

**How it works:** Detects topic changes and splits there.

```bash
sw-search ./docs \
  --chunking-strategy topic \
  --topic-threshold 0.2
```

**Parameters:**

- `topic-threshold`: Sensitivity to topic changes (lower = more splits)

**Best for:**

- Long-form content covering multiple topics
- Articles that shift between subjects
- Meeting transcripts

**When to avoid:** Focused documents on a single topic.

### 7. QA-Optimized

**How it works:** Optimized for question-answering, preserves questions with their answers.

```bash
sw-search ./docs --chunking-strategy qa
```

**Best for:**

- FAQ documents
- Tutorial content with "How do I..." sections
- Documentation with Q&A structure
- Troubleshooting guides

**Features:**

- Detects question patterns (what, how, why, etc.)
- Keeps questions with surrounding context
- Adds metadata: `has_question`, `has_process`

**Example:**

```
Text:
"How do you create an agent? First, inherit from AgentBase. Then define your configuration. Finally, call run() to start the server."

Chunk metadata: {has_question: true, has_process: true}
→ Preserved as single chunk with full context
```

### 8. Markdown-Aware

**How it works:** Chunks at header boundaries, detects code blocks, preserves structure.

```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --file-types md
```

**Best for:**

- Technical documentation with code examples
- GitHub README files
- API documentation
- Developer guides

**Features:**

- Chunks at `##` header boundaries
- Never splits code blocks
- Detects programming language in code blocks
- Adds tags: `code`, `code:python`, `code:bash`
- Preserves header hierarchy in metadata

This strategy is so important we'll cover it in detail in the next section.

### 9. JSON (Pre-chunked Content)

**How it works:** Reads pre-chunked content from JSON files.

```bash
sw-search ./chunks/ \
  --chunking-strategy json \
  --file-types json
```

**Best for:**

- Manually curated knowledge bases
- Content from APIs that's already structured
- When you need precise control over chunks

**Features:**

- You define the chunks
- You define the metadata
- Full control over structure

We'll cover the JSON workflow in detail in section 7.

## Choosing the Right Strategy

Here's a decision tree:

```
Do you need code examples to be findable?
├─ Yes → Use markdown strategy
└─ No
   ├─ Is it FAQ/tutorial content?
   │  └─ Yes → Use qa strategy
   └─ No
      ├─ Is it a PDF?
      │  └─ Yes → Use page strategy
      └─ No
         ├─ Need precise control?
         │  └─ Yes → Use json strategy
         └─ No
            └─ Use sentence strategy (default)
```

## Comparing Strategies on Real Content

Let's compare strategies on the same documentation:

**Original text (API docs):**

```markdown
## Authentication

All API requests require authentication. Use your API key in the Authorization header.

### Python Example


import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY"
}
response = requests.get("https://api.example.com/data", headers=headers)


### Common Errors

Error 401 means invalid credentials. Error 403 means insufficient permissions.
```

**Sentence strategy:**

- Chunk 1: "All API requests require authentication. Use your API key in the Authorization header."
- Chunk 2: "import requests headers = { "Authorization": "Bearer YOUR_API_KEY" } response = ..."
- Chunk 3: "Error 401 means invalid credentials. Error 403 means insufficient permissions."

**Problem:** Code split from context, hard to read.

**Markdown strategy:**

- Chunk 1 (## Authentication section):
  - Content: Full section including header and description
  - Metadata: {h2: "Authentication"}
- Chunk 2 (### Python Example section):
  - Content: Header + complete code block (unsplit)
  - Metadata: {h2: "Authentication", h3: "Python Example", has_code: true, tags: ["code", "code:python"]}
- Chunk 3 (### Common Errors section):
  - Content: Header + error descriptions
  - Metadata: {h2: "Authentication", h3: "Common Errors"}

**Better:** Code intact, hierarchical context preserved, searchable by code tag.

## Chunk Size Recommendations

General guidelines:

- **Sentence strategy:** 5-8 sentences per chunk
- **Paragraph strategy:** Let natural paragraphs define size
- **Sliding window:** 100-200 words, 20-40 word overlap
- **Markdown strategy:** Let headers define size (naturally varies)

## Testing Your Strategy

Build test indexes with different strategies:

```bash
sw-search ./docs --chunking-strategy sentence --output sentence.swsearch
sw-search ./docs --chunking-strategy markdown --output markdown.swsearch
sw-search ./docs --chunking-strategy qa --output qa.swsearch
```

Then test queries:

```bash
sw-search search sentence.swsearch "how to authenticate"
sw-search search markdown.swsearch "how to authenticate"
sw-search search qa.swsearch "how to authenticate"
```

Compare which returns the most useful results.

## Inspecting Chunks

Export chunks to JSON to see how they're split:

```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --output-format json \
  --output chunks.json
```

Open `chunks.json` and review:
- Are chunks too small? (missing context)
- Are chunks too large? (too much noise)
- Are code blocks split? (bad)
- Are related concepts separated? (bad)

Adjust your strategy based on what you see.

## Key Takeaways

1. **Chunking impacts search quality** - poor chunks = poor results
2. **Different content needs different strategies** - one size doesn't fit all
3. **Markdown strategy is best for technical docs** - preserves code and structure
4. **QA strategy for FAQ content** - optimized for questions and answers
5. **Default (sentence) works for general content** - good starting point
6. **Test and compare** - build test indexes and query them
7. **Inspect chunks** - export to JSON to see what you're creating

In the next section, we'll deep-dive into the markdown strategy, which is specifically designed for technical documentation with code examples.
# 6. The Markdown Strategy: Perfect for Technical Documentation

If you're building an agent to answer questions about technical documentation, the markdown chunking strategy is your best choice. It's specifically designed to handle what generic chunking strategies get wrong: code blocks and document structure.

## The Problem with Generic Chunking

Generic chunking strategies (sentence, paragraph) treat documentation as plain text. They don't understand the structure of technical docs:

**What they miss:**

- Code blocks are meaningful units that shouldn't be split
- Headers indicate topic boundaries
- Code language matters (Python vs JavaScript)
- Section hierarchy provides context
- Code examples are high-value content

**What happens:**

```python
# Generic chunking might do this:
Chunk 1: "...configure your agent using set_params(). Here's an example: ```python\nag"
Chunk 2: "ent.set_params({'temperature': 0.7})```\n\nThis configures the temperature..."
```

The code block is split across chunks. Neither chunk is useful.

## What the Markdown Strategy Does Differently

The markdown strategy understands technical documentation:

### 1. Chunks at Header Boundaries

Headers (`##`, `###`, etc.) mark logical topic changes. The strategy uses them as natural chunk boundaries:

```markdown
## Authentication

Content about authentication...

## Configuration

Content about configuration...
```

Becomes two chunks:
- Chunk 1: "Authentication" section (complete)
- Chunk 2: "Configuration" section (complete)

### 2. Never Splits Code Blocks

Code blocks are detected and treated as atomic units:

```markdown
## Example

Here's how to create an agent:


from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="MyAgent")


This creates a basic agent.
```

Becomes one chunk with the complete code block intact.

### 3. Detects Programming Language

The strategy extracts the language from code fence syntax:

```markdown

# Python code



# Bash code



// JavaScript code

```

Each chunk gets language-specific metadata and tags.

### 4. Adds Rich Metadata

Each chunk automatically gets metadata:

```python
{
    "h1": "Getting Started",           # Top-level header
    "h2": "Creating Your First Agent", # Second-level header
    "h3": "Python Example",            # Third-level header
    "has_code": True,                  # Contains code blocks
    "code_languages": ["python"],      # Languages in this chunk
    "tags": ["code", "code:python"],   # Searchable tags
    "depth": 3                         # Header nesting depth
}
```

This metadata powers hybrid search boosting.

### 5. Preserves Header Hierarchy

Each chunk knows its place in the document structure:

```markdown
# API Reference

## Authentication

### Bearer Tokens

Content here...
```

The "Bearer Tokens" chunk has full context:
- h1: "API Reference"
- h2: "Authentication"
- h3: "Bearer Tokens"

When this chunk appears in search results, the agent knows the full context path.

## How Code Tags Boost Search

Remember the hybrid search algorithm? Code chunks get special treatment:

**Query:** "show me python code for agent configuration"

**Without markdown strategy:**

- Generic chunks might not identify code blocks
- No "code" or "code:python" tags
- Code examples don't get boosted
- Results might return prose about code instead of actual code

**With markdown strategy:**

- Code blocks detected automatically
- Tagged with "code" and "code:python"
- Hybrid search applies 20% boost for code chunks (when keywords match)
- Actual code examples rise to the top

This is why the markdown strategy is crucial for technical documentation.

## Real-World Example

Let's walk through a complete example:

**Source documentation (voice_config.md):**

```markdown
# Voice Configuration

## Overview

The SignalWire Agents SDK provides flexible voice configuration options.

## Setting Voice Parameters

Use the `add_language()` method to configure voice settings.

### Python Example


agent.add_language(
    name="English",
    code="en-US",
    voice="elevenlabs.adam",
    function_fillers=["One moment...", "Let me check..."]
)


### JavaScript Example


agent.addLanguage({
    name: "English",
    code: "en-US",
    voice: "elevenlabs.adam"
});


## Available Voices

SignalWire supports multiple voice providers:
- ElevenLabs voices (elevenlabs.*)
- OpenAI voices (openai.*)
- Google voices (google.*)

## Common Issues

If voice is not working, check that your voice provider is configured correctly.
```

**Chunks created:**

**Chunk 1: Overview**
```
Section: Voice Configuration > Overview
Content: "The SignalWire Agents SDK provides flexible voice configuration options."
Metadata: {
    h1: "Voice Configuration",
    h2: "Overview",
    depth: 2
}
```

**Chunk 2: Setting Voice Parameters + Python Example**
```
Section: Voice Configuration > Setting Voice Parameters > Python Example
Content: "Use the `add_language()` method to configure voice settings.\n\n### Python Example\n\n```python\nagent.add_language(...)```"
Metadata: {
    h1: "Voice Configuration",
    h2: "Setting Voice Parameters",
    h3: "Python Example",
    has_code: true,
    code_languages: ["python"],
    tags: ["code", "code:python"],
    depth: 3
}
```

**Chunk 3: JavaScript Example**
```
Section: Voice Configuration > Setting Voice Parameters > JavaScript Example
Content: "### JavaScript Example\n\n```javascript\nagent.addLanguage(...)```"
Metadata: {
    h1: "Voice Configuration",
    h2: "Setting Voice Parameters",
    h3: "JavaScript Example",
    has_code: true,
    code_languages: ["javascript"],
    tags: ["code", "code:javascript"],
    depth: 3
}
```

**Chunk 4: Available Voices**
```
Section: Voice Configuration > Available Voices
Content: "SignalWire supports multiple voice providers:\n- ElevenLabs voices (elevenlabs.*)\n- OpenAI voices (openai.*)\n- Google voices (google.*)"
Metadata: {
    h1: "Voice Configuration",
    h2: "Available Voices",
    depth: 2
}
```

**Chunk 5: Common Issues**
```
Section: Voice Configuration > Common Issues
Content: "If voice is not working, check that your voice provider is configured correctly."
Metadata: {
    h1: "Voice Configuration",
    h2: "Common Issues",
    depth: 2
}
```

## Search Query Examples

Now let's see how these chunks perform in search:

**Query:** "python code for configuring voice"

**Matching:**

- Chunk 2 has strong vector similarity (configuration + voice)
- Keywords match: "python", "code", "configuring", "voice"
- Has tags: "code", "code:python"
- Hybrid boost: ~40% (keywords + code tag)

**Result:** Chunk 2 ranks #1, user gets exact code they need.

---

**Query:** "what voice providers are supported"

**Matching:**

- Chunk 4 has strong vector similarity (providers + supported)
- Keywords match: "voice", "providers"
- Metadata boost from header context

**Result:** Chunk 4 ranks #1, user gets provider list.

---

**Query:** "voice not working troubleshooting"

**Matching:**

- Chunk 5 has strong vector similarity (working + troubleshooting)
- Keywords match: "voice", "working"
- h2: "Common Issues" is relevant

**Result:** Chunk 5 ranks #1, user gets troubleshooting info.

## Language-Specific Queries

The code language tags enable precise searches:

**Query:** "javascript example voice"

- Chunk 3 matches: has "code:javascript" tag
- Chunk 2 excluded: has "code:python" tag (not JavaScript)

Users get language-specific examples.

## Building with Markdown Strategy

```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --file-types md \
  --output docs.swsearch
```

The strategy automatically:
- Scans for markdown files
- Parses header structure
- Detects code blocks with language
- Generates metadata
- Creates chunks at logical boundaries
- Tags code chunks appropriately

## Best Practices

### 1. Use Clear Headers

Good header structure helps chunking:

```markdown
## Topic (h2 for main topics)
### Subtopic (h3 for subtopics)
#### Detail (h4 for details)
```

Avoid skipping levels:
```markdown
## Topic
#### Detail (bad - skipped h3)
```

### 2. Keep Code Examples Under Headers

```markdown
### Python Example


# Code here

```

This ensures code chunks have descriptive headers in their metadata.

### 3. Use Language Tags in Code Fences

Always specify the language:

```markdown
  ← Good
# code


          ← Bad (no language)
# code

```

Without language tags, code chunks miss language-specific metadata.

### 4. Organize by Logical Sections

Each header should represent a complete, self-contained topic:

```markdown
## Authentication (complete topic)
### API Keys (subtopic)
### OAuth (subtopic)

## Rate Limiting (new complete topic)
```

### 5. Don't Bury Code in Prose

Make code examples distinct sections:

**Good:**
```markdown
## Example

Here's how to authenticate:

### Code


# code

```

**Bad:**
```markdown
## Example

Here's how to authenticate: 
# code
 and that's it.
```

## Limitations

The markdown strategy won't help with:
- Non-markdown content (use appropriate strategy)
- Poorly structured markdown (missing headers)
- Code outside fenced blocks (inline code is just text)
- Documentation that doesn't use headers

For such content, consider the JSON workflow where you manually structure the chunks.

## Combining with pgvector

The markdown strategy works with all backends:

**SQLite (local .swsearch file):**
```bash
sw-search ./docs --chunking-strategy markdown --output docs.swsearch
```

**pgvector (PostgreSQL):**
```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --backend pgvector \
  --connection-string "postgresql://user:pass@localhost:5432/knowledge" \
  --output signalwire_docs
```

Same chunking, different storage.

## Inspecting Markdown Chunks

Export to JSON to see the structure:

```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --output-format json \
  --output chunks.json
```

Review the chunks to verify:
- ✅ Code blocks are intact
- ✅ Headers are captured in metadata
- ✅ Languages are detected
- ✅ Tags are present
- ✅ Chunk sizes are reasonable

## Key Takeaways

1. **Markdown strategy understands technical docs** - headers, code, structure
2. **Code blocks never split** - preserved as atomic units
3. **Automatic language detection** - tags like "code:python"
4. **Header hierarchy preserved** - full context path in metadata
5. **Code chunks get boosted** - 20% boost in hybrid search
6. **Best for developer documentation** - API docs, README files, guides
7. **Use clear header structure** - helps chunking quality

Next, we'll explore the JSON workflow for when you need even more control over your chunks - manually curating and structuring your knowledge base.
# 7. The JSON Workflow: Manual Curation

Sometimes automatic chunking isn't enough. You need precise control over what gets indexed, how it's structured, and what metadata is attached. The JSON workflow gives you that control.

## Why Manual Curation?

Automatic chunking works well for most documentation, but there are cases where you need human judgment:

**When to use the JSON workflow:**

1. **High-value knowledge bases** where accuracy is critical
2. **API documentation** scraped from structured sources
3. **Content from databases** that's already structured
4. **Complex relationships** between chunks (TOC entries, related content)
5. **Custom metadata** that can't be derived automatically
6. **Quality control** - review and approve every chunk

**Example scenario:** You're building a customer support agent. The knowledge base includes:
- Product documentation
- Troubleshooting guides
- Known issues with workarounds
- Links to related articles

You want to ensure each chunk has the right metadata, related links are preserved, and content is organized by priority. Manual curation lets you do this.

## The Two-Phase Workflow

The JSON workflow has two distinct phases:

### Phase 1: Export Chunks to JSON

First, use automatic chunking to generate candidate chunks, then export them to JSON for review:

```bash
# Export all chunks to a single JSON file
sw-search ./docs \
  --chunking-strategy markdown \
  --output-format json \
  --output all_chunks.json
```

Or export to separate files (one per source document):

```bash
# Export to directory with one JSON file per source
sw-search ./docs \
  --chunking-strategy markdown \
  --output-format json \
  --output-dir ./chunks/
```

This creates JSON files with the full chunk structure that you can edit.

### Phase 2: Build Index from JSON

After reviewing and editing the JSON, build the final index:

```bash
sw-search ./chunks/ \
  --chunking-strategy json \
  --file-types json \
  --output final.swsearch
```

The `json` chunking strategy reads pre-structured chunks from JSON files instead of processing raw documents.

## JSON Format Structure

The JSON format is straightforward:

```json
{
  "chunks": [
    {
      "chunk_id": "auth_overview_1",
      "type": "content",
      "content": "All API requests require authentication using Bearer tokens.",
      "metadata": {
        "section": "Authentication",
        "category": "api",
        "tags": ["authentication", "api", "security"],
        "url": "https://docs.example.com/auth",
        "priority": "high",
        "related_chunks": ["auth_example_1", "auth_errors_1"]
      }
    },
    {
      "chunk_id": "auth_example_1",
      "type": "content",
      "content": "Example: Authorization: Bearer YOUR_API_KEY",
      "metadata": {
        "section": "Authentication",
        "category": "example",
        "tags": ["authentication", "example", "code"],
        "url": "https://docs.example.com/auth#examples",
        "language": "http",
        "related_chunks": ["auth_overview_1"]
      }
    },
    {
      "chunk_id": "toc_auth",
      "type": "toc",
      "content": "Authentication",
      "metadata": {
        "section_number": 1,
        "related_toc": "toc_main",
        "tags": ["toc", "navigation"]
      }
    }
  ]
}
```

### Required Fields

- **content** (string): The actual text content of the chunk
- **chunks** (array): Top-level array containing all chunks

### Optional Fields

- **chunk_id** (string): Unique identifier for this chunk
- **type** (string): "content" or "toc" (table of contents)
- **metadata** (object): Any custom metadata you want

### Metadata Fields (All Optional)

You can include any metadata you want, but here are common fields:

```json
{
  "section": "Section name",
  "category": "Category for organization",
  "tags": ["tag1", "tag2"],
  "url": "Source URL",
  "priority": "high|medium|low",
  "difficulty": "beginner|intermediate|advanced",
  "related_chunks": ["id1", "id2"],
  "related_toc": "toc_id",
  "section_number": 1,
  "language": "python",
  "has_code": true,
  "code_languages": ["python", "bash"]
}
```

The metadata becomes searchable and can boost results in hybrid search.

## Phase 1 Example: Export and Review

Let's export some documentation:

```bash
sw-search ./docs/api.md \
  --chunking-strategy markdown \
  --output-format json \
  --output api_chunks.json
```

This creates `api_chunks.json`:

```json
{
  "chunks": [
    {
      "chunk_id": "chunk_0",
      "type": "content",
      "content": "# API Reference\n\nThe SignalWire API provides programmatic access to all platform features.",
      "metadata": {
        "chunk_method": "markdown",
        "chunk_index": 0,
        "h1": "API Reference",
        "filename": "api.md",
        "tags": []
      }
    },
    {
      "chunk_id": "chunk_1",
      "type": "content",
      "content": "## Authentication\n\nAll requests require authentication...",
      "metadata": {
        "chunk_method": "markdown",
        "chunk_index": 1,
        "h1": "API Reference",
        "h2": "Authentication",
        "filename": "api.md",
        "tags": []
      }
    }
  ]
}
```

Now you can review and edit this JSON.

## Phase 2 Example: Edit and Enhance

Open `api_chunks.json` and make improvements:

**1. Add descriptive chunk IDs:**

```json
{
  "chunk_id": "api_authentication_overview",  // Was: chunk_1
  ...
}
```

**2. Add relevant tags:**

```json
{
  "metadata": {
    "tags": ["authentication", "security", "api", "getting-started"],
    ...
  }
}
```

**3. Add URLs to source documentation:**

```json
{
  "metadata": {
    "url": "https://docs.signalwire.com/api#authentication",
    ...
  }
}
```

**4. Add priority levels:**

```json
{
  "metadata": {
    "priority": "high",  // Critical info
    ...
  }
}
```

**5. Link related chunks:**

```json
{
  "metadata": {
    "related_chunks": [
      "api_authentication_example",
      "api_authentication_errors"
    ],
    ...
  }
}
```

**6. Remove low-value chunks:**

Delete chunks that are just headers with no useful content:

```json
{
  "chunks": [
    // Removed chunk_0 (just the h1 header)
    {
      "chunk_id": "api_authentication_overview",
      ...
    }
  ]
}
```

**7. Merge related chunks:**

If two chunks are always relevant together, merge them:

```json
{
  "chunk_id": "api_authentication_complete",
  "content": "## Authentication\n\nAll requests require authentication...\n\n### Example\n\nAuthorization: Bearer YOUR_API_KEY",
  "metadata": {
    "tags": ["authentication", "example", "code"],
    "combined": true
  }
}
```

**8. Add custom metadata:**

```json
{
  "metadata": {
    "difficulty": "beginner",
    "estimated_time": "5 minutes",
    "prerequisites": ["account_creation"],
    ...
  }
}
```

## Phase 3: Build the Final Index

After editing, build the index:

```bash
sw-search ./api_chunks.json \
  --chunking-strategy json \
  --file-types json \
  --output api.swsearch
```

The `json` strategy reads your curated chunks and creates the search index.

## Creating JSON from Scratch

You can also create JSON chunks programmatically or manually, without exporting first:

```python
import json

chunks = {
    "chunks": [
        {
            "chunk_id": "welcome",
            "type": "content",
            "content": "Welcome to SignalWire! This guide will help you get started.",
            "metadata": {
                "section": "Getting Started",
                "tags": ["welcome", "introduction"],
                "priority": "high"
            }
        },
        {
            "chunk_id": "quickstart",
            "type": "content",
            "content": "Quick start: pip install signalwire-agents",
            "metadata": {
                "section": "Getting Started",
                "tags": ["installation", "quickstart"],
                "related_chunks": ["welcome"]
            }
        }
    ]
}

with open("manual_chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)
```

Then build the index:

```bash
sw-search manual_chunks.json \
  --chunking-strategy json \
  --file-types json \
  --output manual.swsearch
```

## Use Case: Scraping API Documentation

Let's say you're scraping documentation from an API and want to structure it for search:

```python
import json
import requests
from bs4 import BeautifulSoup

def scrape_api_docs(base_url):
    """Scrape API docs and create structured chunks"""
    chunks = []

    # Fetch API docs (simplified)
    response = requests.get(f"{base_url}/api/docs")
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract endpoints
    for endpoint in soup.find_all('div', class_='endpoint'):
        method = endpoint.find('span', class_='method').text
        path = endpoint.find('span', class_='path').text
        description = endpoint.find('p', class_='description').text

        # Create chunk
        chunk = {
            "chunk_id": f"endpoint_{method}_{path.replace('/', '_')}",
            "type": "content",
            "content": f"{method} {path}\n\n{description}",
            "metadata": {
                "section": "API Endpoints",
                "method": method,
                "path": path,
                "tags": ["api", "endpoint", method.lower()],
                "url": f"{base_url}/api/docs#{path}"
            }
        }
        chunks.append(chunk)

    return {"chunks": chunks}

# Scrape and save
docs = scrape_api_docs("https://api.example.com")
with open("api_chunks.json", "w") as f:
    json.dump(docs, f, indent=2)
```

Then build the index as usual.

## Use Case: Table of Contents

Create TOC entries that help navigate to detailed content:

```json
{
  "chunks": [
    {
      "chunk_id": "toc_main",
      "type": "toc",
      "content": "Table of Contents: Getting Started, API Reference, Guides",
      "metadata": {
        "tags": ["toc", "navigation"],
        "section_number": 0
      }
    },
    {
      "chunk_id": "toc_getting_started",
      "type": "toc",
      "content": "Getting Started: Installation, Quick Start, First Agent",
      "metadata": {
        "tags": ["toc", "getting-started"],
        "related_toc": "toc_main",
        "section_number": 1
      }
    },
    {
      "chunk_id": "content_installation",
      "type": "content",
      "content": "Installation instructions...",
      "metadata": {
        "related_toc": "toc_getting_started",
        "tags": ["installation", "getting-started"]
      }
    }
  ]
}
```

TOC chunks help users navigate, and the `related_toc` metadata links content to structure.

## Validation and Testing

Before building the final index, validate your JSON:

```python
import json

def validate_chunks(filename):
    """Validate chunk JSON structure"""
    with open(filename) as f:
        data = json.load(f)

    if "chunks" not in data:
        print("ERROR: Missing 'chunks' key")
        return False

    for i, chunk in enumerate(data["chunks"]):
        if "content" not in chunk:
            print(f"ERROR: Chunk {i} missing 'content'")
            return False

        # Check for empty content
        if not chunk["content"].strip():
            print(f"WARNING: Chunk {i} has empty content")

        # Check for recommended fields
        if "chunk_id" not in chunk:
            print(f"WARNING: Chunk {i} missing 'chunk_id'")

        if "metadata" in chunk:
            if "tags" not in chunk["metadata"]:
                print(f"INFO: Chunk {i} has no tags")

    print(f"Validated {len(data['chunks'])} chunks")
    return True

validate_chunks("api_chunks.json")
```

## Comparing Workflows

**Automatic workflow (markdown strategy):**

- ✅ Fast - no manual work
- ✅ Consistent - same logic for all content
- ❌ Less control - automated decisions
- ❌ No custom relationships

**JSON workflow (manual curation):**

- ✅ Full control - human judgment
- ✅ Custom metadata - anything you need
- ✅ Quality assurance - review each chunk
- ❌ Time-consuming - manual editing
- ❌ Maintenance - must update JSON when docs change

**Best of both:**
1. Use automatic chunking for most content
2. Use JSON workflow for critical, high-value sections
3. Combine both in a single index

## Combining Both Approaches

You can mix automatic and manual chunks:

```bash
# Export critical docs to JSON for curation
sw-search ./docs/critical/ \
  --output-format json \
  --output critical_chunks.json

# Edit critical_chunks.json manually

# Build index from both curated JSON and automatic chunking
sw-search ./critical_chunks.json ./docs/other/ \
  --chunking-strategy json \
  --file-types json,md \
  --output combined.swsearch
```

The index contains both hand-curated and automatically chunked content.

## Key Takeaways

1. **JSON workflow provides full control** - manual curation for quality
2. **Two-phase process** - export, edit, rebuild
3. **Simple JSON format** - chunks array with content and metadata
4. **Flexible metadata** - add any fields you need
5. **Great for high-value content** - worth the manual effort
6. **Can combine with automatic chunking** - best of both worlds
7. **Programmatic generation** - scrape APIs, structure databases
8. **TOC and relationships** - link related content

Next, we'll explore the different embedding models available and how to choose between speed and quality based on your needs.
# 8. Search Models: Performance vs Quality

The embedding model you choose affects both search quality and performance. Understanding the tradeoffs helps you pick the right model for your use case.

## Available Models

The SDK provides three model aliases that map to specific sentence-transformers models:

### Mini (Recommended Default)

```bash
sw-search ./docs --model mini
```

**Full model name:** `sentence-transformers/all-MiniLM-L6-v2`

**Characteristics:**

- **Dimensions:** 384
- **Speed:** Very fast (~5x faster than base)
- **Size:** ~80MB
- **Quality:** Excellent for most use cases
- **Memory:** Low usage

**When to use:**

- General documentation
- Fast query responses required
- Memory-constrained environments
- Large indexes (millions of chunks)
- Real-time search in production

**Performance:**

- Embedding generation: ~500 chunks/second (CPU)
- Query time: <10ms for 10,000 chunks
- Memory: ~200MB for 100,000 chunks

### Base

```bash
sw-search ./docs --model base
```

**Full model name:** `sentence-transformers/all-mpnet-base-v2`

**Characteristics:**

- **Dimensions:** 768
- **Speed:** Moderate (baseline reference)
- **Size:** ~420MB
- **Quality:** High quality, good for complex queries
- **Memory:** Moderate usage

**When to use:**

- Complex technical documentation
- Nuanced semantic understanding needed
- When quality matters more than speed
- Smaller indexes (<100k chunks)

**Performance:**

- Embedding generation: ~100 chunks/second (CPU)
- Query time: ~20ms for 10,000 chunks
- Memory: ~400MB for 100,000 chunks

### Large

```bash
sw-search ./docs --model large
```

**Full model name:** Currently maps to `all-mpnet-base-v2` (same as base)

**Note:** This alias is reserved for future larger models. Currently equivalent to base.

## Speed vs Quality Comparison

Let's compare real-world performance:

**Test setup:**

- 10,000 documentation chunks
- Single query: "how do I configure voice settings"
- Hardware: Standard CPU (no GPU)

**Results:**

| Model | Build Time | Index Size | Query Time | Quality Score |
|-------|------------|------------|------------|---------------|
| mini  | 20 seconds | 15 MB      | 8 ms       | 0.92          |
| base  | 100 seconds| 30 MB      | 18 ms      | 0.95          |

**Interpretation:**

- Mini is 5x faster to build
- Mini index is 50% smaller
- Mini queries are 2x faster
- Base has ~3% better quality

For most documentation, that 3% quality difference is negligible. The 5x speed improvement is significant.

## Understanding Embedding Dimensions

Embeddings are vectors - lists of numbers representing meaning. The dimension count is how many numbers:

**384 dimensions (mini):**
```
[0.23, -0.15, 0.89, 0.04, ..., 0.67]  // 384 numbers
```

**768 dimensions (base):**
```
[0.23, -0.15, 0.89, 0.04, ..., 0.67]  // 768 numbers
```

More dimensions = more capacity to capture subtle semantic nuances.

**Analogy:** Think of GPS coordinates:
- 2D (latitude/longitude) locates you on Earth's surface
- 3D (+ altitude) adds vertical position
- More dimensions = more precise location in "meaning space"

But more dimensions also means:
- Slower computation (more numbers to process)
- More memory (larger vectors to store)
- Diminishing returns (768 vs 384 matters less than 384 vs 100)

## When Quality Differences Matter

The mini model is excellent for most cases. When might you need base?

### Subtle Semantic Distinctions

When your content has nuanced differences:

**Example:**
```
Doc 1: "The set_params() method configures agent parameters"
Doc 2: "The set_params() function modifies agent settings"
Doc 3: "Use set_params() to customize agent behavior"
```

These are very similar. Base model might better distinguish which is most relevant for:
- Query: "how to change agent configuration" (→ Doc 2)
- Query: "method for agent parameters" (→ Doc 1)
- Query: "customizing agent" (→ Doc 3)

Mini model would likely rate all three similarly (which is often fine).

### Domain-Specific Language

Technical jargon with subtle differences:

**Example:**
```
"latency" vs "lag" vs "delay"
"authenticate" vs "authorize" vs "validate"
"compile" vs "build" vs "transpile"
```

Base model has slightly better understanding of these distinctions.

### Multi-Language Content

If your documentation mixes languages:

```
"Le SDK SignalWire permet de créer des agents IA"
"The SignalWire SDK enables creating AI agents"
```

Base model handles cross-language semantic similarity slightly better.

## When Mini is Sufficient (Most Cases)

For typical technical documentation, mini is excellent:

**Clear content with good structure:**
```markdown
## Installation

Install the package:

pip install signalwire-agents


## Configuration

Configure your agent:

agent = AgentBase(name="MyAgent")

```

There's no ambiguity here. Mini finds this perfectly.

**Well-written documentation:**

- Clear headings
- Distinct topics per section
- Code examples
- Consistent terminology

→ Mini works great

**Conversational queries:**

- "how do I install"
- "show me configuration example"
- "what is the AgentBase class"

→ Mini handles these well

## Using Custom Models

You can use any sentence-transformers model:

```bash
sw-search ./docs --model sentence-transformers/all-MiniLM-L12-v2
```

**Popular alternatives:**

**Multi-lingual models:**
```bash
sw-search ./docs --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Instruction-tuned models:**
```bash
sw-search ./docs --model hkunlp/instructor-base
```

**Domain-specific models:**
```bash
sw-search ./docs --model pritamdeka/S-PubMedBert-MS-MARCO  # Medical
sw-search ./docs --model sentence-transformers/allenai-specter  # Scientific
```

**Requirements:**

- Must be available on HuggingFace
- Must be sentence-transformers compatible
- Model downloads automatically on first use

## Model Download and Caching

Models download automatically:

```bash
sw-search ./docs --model mini
# First run:
# Downloading model... 80MB
# [████████████████████] 100%
# Model cached at ~/.cache/torch/sentence_transformers/

# Subsequent runs:
# Loading cached model...
# ✓ Ready
```

The model is cached locally. Future builds use the cached version.

**Cache location:**

- Linux/Mac: `~/.cache/torch/sentence_transformers/`
- Windows: `C:\Users\<user>\.cache\torch\sentence_transformers

## Memory Considerations

Model memory usage during indexing:

| Model | Model Size | Peak Memory (10k chunks) |
|-------|------------|--------------------------|
| mini  | 80 MB      | ~500 MB                  |
| base  | 420 MB     | ~1.2 GB                  |

**For large indexes:**

If you're indexing 1M+ chunks, memory matters:

```bash
# Process in batches to reduce memory
sw-search ./docs \
  --model mini \
  --batch-size 1000 \
  --output large.swsearch
```

Mini's lower memory footprint helps with large-scale indexing.

## Query Performance

Query performance scales with dimensions:

**10,000 chunks:**

- mini: ~8ms per query
- base: ~18ms per query

**100,000 chunks:**

- mini: ~80ms per query
- base: ~180ms per query

**1,000,000 chunks:**

- mini: ~800ms per query
- base: ~1.8s per query

For real-time voice agents, query speed matters. Mini's 2x speedup is significant.

## Production Recommendations

**Default choice: mini**

- Start here unless you have specific needs
- Excellent quality-to-speed ratio
- Works for 95% of use cases

**Use base if:**

- You have very subtle semantic distinctions
- Quality is paramount, speed is secondary
- Small index (< 10k chunks) so speed doesn't matter
- You've tested and confirmed better results

**Use custom models if:**

- Multi-language content (use multi-lingual model)
- Domain-specific content (medical, legal, scientific)
- Special requirements (privacy, on-premises, etc.)

## Testing Models

Compare models on your content:

```bash
# Build with mini
sw-search ./docs --model mini --output mini.swsearch

# Build with base
sw-search ./docs --model base --output base.swsearch

# Test queries
sw-search search mini.swsearch "your test query" --verbose
sw-search search base.swsearch "your test query" --verbose
```

Compare:
- Which returns more relevant results?
- Does base provide better results worth the slowdown?
- Is the quality difference noticeable?

Usually mini wins on the speed/quality tradeoff.

## Mixing Models is Not Recommended

Don't mix models in a single index:

**Bad:**
```bash
# Build part of index with mini
sw-search ./docs1 --model mini --output partial.swsearch

# Try to add more with base
sw-search ./docs2 --model base --append partial.swsearch  # ❌ Don't do this
```

Embeddings from different models aren't comparable. Stick with one model per index.

## Migrating Between Models

To change models, rebuild the index:

```bash
# Original index (base model)
sw-search ./docs --model base --output docs_base.swsearch

# Switch to mini model
sw-search ./docs --model mini --output docs_mini.swsearch
```

You can't convert embeddings between models. Re-processing is required.

## Key Takeaways

1. **Start with mini** - best default for most use cases
2. **5x speed advantage** - mini is significantly faster
3. **Minimal quality loss** - ~3% difference on benchmarks
4. **Memory matters** - mini uses less memory at scale
5. **Test on your content** - compare if quality matters
6. **Can't mix models** - stick with one per index
7. **Custom models available** - for specialized needs

Next, we'll explore deployment options: when to use local `.swsearch` files versus scaling to PostgreSQL pgvector for production multi-agent systems.
# 9. Deployment Options: SQLite vs pgvector

The SDK supports two storage backends for search indexes: SQLite (via `.swsearch` files) and PostgreSQL with the pgvector extension. Understanding when to use each is critical for production deployments.

## SQLite Backend (.swsearch Files)

The default backend stores everything in a single portable file.

### How .swsearch Files Work

A `.swsearch` file is a SQLite database containing:
- Vector embeddings for all chunks
- Original text content
- Metadata and tags
- Search configuration
- Model information

**Everything in one file:**
```bash
ls -lh knowledge.swsearch
# -rw-r--r--  1 user  staff   2.3M  knowledge.swsearch
```

Deploy it with your agent, and search works immediately.

### Advantages of .swsearch Files

**1. Portability**

A single file contains your entire knowledge base:
```bash
# Build on dev machine
sw-search ./docs --output knowledge.swsearch

# Copy to production
scp knowledge.swsearch prod:/app/

# Agent loads it directly
agent.add_skill("native_vector_search", {
    "index_path": "./knowledge.swsearch"
})
```

No database setup. No connection strings. Just a file.

**2. Simplicity**

Perfect for:
- Development and testing
- Single-agent deployments
- Small to medium knowledge bases (< 1M chunks)
- Serverless deployments (AWS Lambda, Cloud Functions)
- Edge deployments (containers, embedded systems)

**3. Version Control**

Treat knowledge bases like code artifacts:
```bash
git add knowledge.swsearch
git commit -m "Update documentation index"
git push
```

Different versions for different environments:
```
knowledge_v1.swsearch
knowledge_v2.swsearch
knowledge_production.swsearch
knowledge_staging.swsearch
```

**4. No External Dependencies**

SQLite is embedded. No server to run:
- No PostgreSQL installation
- No connection pooling
- No authentication
- No network calls

**5. Fast for Single Users**

SQLite is optimized for single-user access:
- Low latency (local file)
- Efficient for reads
- No network overhead

### Limitations of .swsearch Files

**1. No Concurrent Writes**

SQLite doesn't handle multiple writers well:
- One agent writing to the index works fine
- Multiple agents trying to update the same index causes contention
- Read-only access is fine for multiple agents

**2. No Live Updates**

To update the index, you must rebuild the entire file:
```bash
# Can't incrementally add to existing index
sw-search ./new_docs --append knowledge.swsearch  # Not supported

# Must rebuild entire index
sw-search ./all_docs --output knowledge.swsearch
```

**3. File Size Limits**

While SQLite can handle large files, there are practical limits:
- 1M chunks: ~500MB file (manageable)
- 10M chunks: ~5GB file (getting large)
- 100M+ chunks: Consider pgvector

Large files mean:
- Slower deployments (copying gigabytes)
- More memory at startup (loading indexes)
- Longer rebuild times

**4. No Multi-Collection Management**

One file = one collection. For multiple knowledge bases:
```python
# Must manage multiple files
agent.add_skill("native_vector_search", {
    "index_path": "./docs.swsearch",
    "tool_name": "search_docs"
})

agent.add_skill("native_vector_search", {
    "index_path": "./api.swsearch",
    "tool_name": "search_api"
})
```

This works, but each file loads separately.

## PostgreSQL pgvector Backend

For production multi-agent systems, pgvector provides enterprise-grade vector storage.

### What is pgvector?

pgvector is a PostgreSQL extension that adds vector data types and similarity search:

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB
);

CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
```

It turns PostgreSQL into a vector database.

### Advantages of pgvector

**1. Multi-Agent Concurrent Access**

Multiple agents query the same database:
```
Agent 1 ──┐
Agent 2 ──┼──> PostgreSQL (pgvector)
Agent 3 ──┘
```

No file contention. No duplication. Shared knowledge base.

**2. Multiple Collections**

Organize knowledge bases as separate collections:
```sql
Collections in database:
- signalwire_unified (main docs)
- pricing (pricing info)
- freeswitch (telephony docs)
- internal_kb (private knowledge)
```

**Example from production (Sigmond agent):**
```python
# Search SignalWire docs
self.add_skill("native_vector_search", {
    "tool_name": "search_signalwire_knowledge",
    "backend": "pgvector",
    "connection_string": pgvector_connection,
    "collection_name": "signalwire_unified"
})

# Search pricing info
self.add_skill("native_vector_search", {
    "tool_name": "search_pricing",
    "backend": "pgvector",
    "connection_string": pgvector_connection,
    "collection_name": "pricing"
})

# Search FreeSWITCH docs
self.add_skill("native_vector_search", {
    "tool_name": "search_freeswitch_knowledge",
    "backend": "pgvector",
    "connection_string": pgvector_connection,
    "collection_name": "freeswitch"
})
```

Three separate knowledge bases, one database, accessed by the same agent.

**3. Incremental Updates**

Update the index without rebuilding everything:
```bash
# Add new documents to existing collection
sw-search ./new_docs \
  --backend pgvector \
  --connection-string "postgresql://..." \
  --collection-name docs \
  --append
```

**4. Scalability**

PostgreSQL scales to billions of rows:
- Efficient indexes (IVFFlat, HNSW)
- Partitioning support
- Replication for high availability
- Backup and restore tools

**5. Enterprise Features**

Standard PostgreSQL capabilities:
- Authentication and authorization
- Connection pooling
- Monitoring and logging
- Backup strategies
- Point-in-time recovery

### Setting Up pgvector

**1. Install PostgreSQL with pgvector:**

```bash
# Ubuntu/Debian
apt install postgresql-15 postgresql-15-pgvector

# Mac with Homebrew
brew install postgresql pgvector

# Or use Docker
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  ankane/pgvector
```

**2. Create database and enable extension:**

```sql
CREATE DATABASE knowledge;
\c knowledge
CREATE EXTENSION vector;
```

**3. Build index to pgvector:**

```bash
sw-search ./docs \
  --backend pgvector \
  --connection-string "postgresql://user:pass@localhost:5432/knowledge" \
  --collection-name signalwire_docs \
  --model mini
```

This creates the collection and populates it with embeddings.

**4. Configure agent to use pgvector:**

```python
import os

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="MyAgent")

        # Build connection string from environment
        pg_user = os.getenv("PGVECTOR_DB_USER", "signalwire")
        pg_pass = os.getenv("PGVECTOR_DB_PASSWORD", "password")
        pg_host = os.getenv("PGVECTOR_HOST", "localhost")
        pg_port = os.getenv("PGVECTOR_PORT", "5432")
        pg_db = os.getenv("PGVECTOR_DB_NAME", "knowledge")

        connection_string = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        # Add search skill
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "backend": "pgvector",
            "connection_string": connection_string,
            "collection_name": "signalwire_docs",
            "model_name": "mini",
            "count": 5
        })
```

### pgvector Performance

**Query performance:**

- 10k chunks: ~15ms
- 100k chunks: ~30ms
- 1M chunks: ~50ms (with proper indexes)

**Indexing options:**

**IVFFlat (default):**
```sql
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
```
- Good balance of speed and accuracy
- Works for millions of vectors

**HNSW (better performance):**
```sql
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);
```
- Faster queries
- More memory usage
- Better for large collections

### Migrating from SQLite to pgvector

The SDK provides a migration tool:

```bash
# Migrate existing .swsearch file to pgvector
sw-search migrate ./knowledge.swsearch \
  --to-pgvector \
  --connection-string "postgresql://user:pass@localhost/knowledge" \
  --collection-name docs
```

This reads the SQLite index and writes it to PostgreSQL.

**Reverse migration (pgvector to SQLite):**

```bash
# Export from pgvector to .swsearch file
sw-search migrate \
  --from-pgvector \
  --connection-string "postgresql://user:pass@localhost/knowledge" \
  --collection-name docs \
  --output knowledge.swsearch
```

Useful for creating portable snapshots of production data.

## When to Use Which Backend

### Use SQLite (.swsearch) When:

- ✅ **Single agent deployment**

- One agent instance
- No concurrent updates
- Simple architecture

- ✅ **Development and testing**

- Fast iteration
- Easy to version control
- No infrastructure needed

- ✅ **Serverless deployments**

- AWS Lambda
- Google Cloud Functions
- Azure Functions
- Package .swsearch with deployment

- ✅ **Edge deployments**

- Embedded systems
- IoT devices
- Offline operation

- ✅ **Small to medium knowledge bases**

- < 1M chunks
- File size < 1GB
- Infrequent updates

- ✅ **Portable agents**

- Demo agents
- Distributable packages
- Self-contained applications

### Use pgvector When:

- ✅ **Multi-agent deployments**

- Multiple agent instances
- Shared knowledge base
- Concurrent queries

- ✅ **Large knowledge bases**

- 1M+ chunks
- Multiple collections
- Frequent updates

- ✅ **Production systems**

- High availability requirements
- Backup and recovery needed
- Monitoring and alerting

- ✅ **Dynamic content**

- Incremental updates
- Real-time indexing
- Content management systems

- ✅ **Multiple knowledge domains**

- Separate collections per domain
- Different access patterns
- Organized knowledge management

## Hybrid Approach

You can use both:

**Development:** Build with .swsearch files
```bash
sw-search ./docs --output dev.swsearch
```

**Staging:** Test with pgvector
```bash
sw-search ./docs \
  --backend pgvector \
  --connection-string "postgresql://localhost/staging" \
  --collection-name docs
```

**Production:** Deploy to pgvector with replicas
```bash
sw-search ./docs \
  --backend pgvector \
  --connection-string "postgresql://prod-db/production" \
  --collection-name docs
```

Same code, different backends via configuration.

## Real-World Architecture: Sigmond

Sigmond uses pgvector with multiple collections:

```
PostgreSQL Database
├── signalwire_unified (5k chunks)
│   └── SDK docs, developer guides, APIs
├── pricing (500 chunks)
│   └── Pricing information, plans
└── freeswitch (2k chunks)
    └── FreeSWITCH telephony docs

Multiple Sigmond Instances
├── Instance 1 ─┐
├── Instance 2 ─┼──> PostgreSQL (shared)
└── Instance 3 ─┘
```

Benefits:
- All instances share same knowledge
- Update once, available everywhere
- Separate tools for different domains
- Centralized knowledge management

## Connection Pooling

For pgvector with many agents, use connection pooling:

**PgBouncer example:**
```ini
[databases]
knowledge = host=postgres-host port=5432 dbname=knowledge

[pgbouncer]
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

Agents connect to PgBouncer instead of directly to PostgreSQL:
```python
connection_string = "postgresql://user:pass@pgbouncer:6432/knowledge"
```

This prevents connection exhaustion with many agents.

## Key Takeaways

1. **SQLite for simplicity** - single file, portable, no setup
2. **pgvector for scale** - multi-agent, concurrent, production-grade
3. **SQLite for dev, pgvector for prod** - hybrid approach works well
4. **Migration tools available** - move between backends as needed
5. **Multiple collections in pgvector** - organize knowledge domains
6. **Connection pooling for pgvector** - handle many concurrent agents
7. **Choose based on deployment** - architecture drives decision

Next, we'll explore the `search-queryonly` install option that dramatically reduces deployment size by excluding model dependencies.
# 10. The search-queryonly Install: Minimizing Dependencies

One of the biggest challenges with vector search is the dependency footprint. The `search-queryonly` install option solves this by separating index building from querying, dramatically reducing production deployment size.

## The Dependency Problem

When you install the full search functionality:

```bash
pip install signalwire-agents[search]
```

You get approximately **500MB** of dependencies, including:

**Heavy ML libraries:**

- `torch` (~200MB) - PyTorch deep learning framework
- `sentence-transformers` (~150MB) - Embedding models
- `transformers` (~100MB) - HuggingFace transformers
- `numpy`, `scipy` (~50MB) - Scientific computing

**Why so large?**

These libraries include:
- Neural network architectures
- Pre-trained model weights
- GPU acceleration support (CUDA libraries)
- Optimization algorithms
- Training infrastructure

All necessary for **generating embeddings** during index building.

## The Key Insight

Here's the critical realization: **production agents don't need to build indexes**.

**Index building (development):**

- Parse documents
- Break into chunks
- **Generate embeddings** ← Needs ML models
- Store in database

**Querying (production):**

- Load pre-computed embeddings
- Compare vectors (just arithmetic)
- Return results

Production agents only need to compare vectors, not generate them. Vector comparison is just multiplication and addition - no ML models required.

## Enter search-queryonly

The `search-queryonly` install gives you just the querying capabilities:

```bash
pip install signalwire-agents[search-queryonly]
```

**Total size:** ~400MB (vs ~500MB for full search)

**What's included:**

- Vector comparison (cosine distance)
- SQLite backend for .swsearch files
- pgvector client for PostgreSQL
- Metadata filtering
- Hybrid search scoring

**What's excluded:**

- PyTorch
- sentence-transformers models
- Embedding generation
- Document processing

**Savings:** ~100MB+ in dependencies

## The Workflow

Separate your development and production environments:

### Development Machine (Build Indexes)

```bash
# Install full search on dev machine
pip install signalwire-agents[search-full]

# Build your indexes
sw-search ./docs --output knowledge.swsearch

# Or build to pgvector
sw-search ./docs \
  --backend pgvector \
  --connection-string "postgresql://localhost/knowledge" \
  --collection-name docs
```

### Production Deployment (Query Only)

```bash
# Install query-only on production
pip install signalwire-agents[search-queryonly]

# Deploy with pre-built index
# Option 1: Copy .swsearch file
scp knowledge.swsearch prod:/app/

# Option 2: Connect to pgvector (embeddings already there)
# Just configure connection string
```

Your production agent queries pre-built indexes without needing ML dependencies.

## Size Comparison

**Container image sizes (example):**

**Full search:**
```dockerfile
FROM python:3.11-slim
RUN pip install signalwire-agents[search]
# Result: ~1.2GB image
```

**Query-only:**
```dockerfile
FROM python:3.11-slim
RUN pip install signalwire-agents[search-queryonly]
COPY knowledge.swsearch /app/
# Result: ~800MB image
```

**Savings:** ~400MB per container

For Kubernetes deployments with multiple replicas:
- 10 replicas × 400MB = 4GB saved
- Faster image pulls
- Less storage costs
- Quicker deployments

## When to Use Query-Only

### Perfect for:

- ✅ **Production deployments**

- Agents only query, don't build
- Pre-built indexes from CI/CD
- Smaller deployment packages

- ✅ **Serverless functions**

- AWS Lambda (250MB unzipped limit)
- Google Cloud Functions
- Azure Functions
- Every MB matters

- ✅ **Edge deployments**

- IoT devices
- Embedded systems
- Resource-constrained environments

- ✅ **Container orchestration**

- Kubernetes clusters
- Docker Swarm
- Multiple replicas
- Reduced storage and bandwidth

- ✅ **CI/CD pipelines**

- Build indexes in CI
- Deploy query-only agents
- Separate concerns

### Not suitable for:

- ❌ **Development environments**

- Need to build/rebuild indexes
- Iterate on chunking strategies
- Test different models

- ❌ **Dynamic indexing**

- Real-time content updates
- User-generated content
- On-the-fly embedding generation

- ❌ **All-in-one deployments**

- Single machine handles everything
- No separation of concerns
- Small-scale deployments where size doesn't matter

## CI/CD Integration

A typical pipeline:

**1. Build Phase (CI server with full dependencies):**

```yaml
# .github/workflows/build-indexes.yml
name: Build Search Indexes

on:
  push:
    paths:
      - 'docs/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install signalwire-agents[search-full]

      - name: Build search index
        run: |
          sw-search ./docs \
            --model mini \
            --chunking-strategy markdown \
            --output knowledge.swsearch

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: search-index
          path: knowledge.swsearch
```

**2. Deploy Phase (production with query-only):**

```yaml
# .github/workflows/deploy.yml
name: Deploy Agent

on:
  workflow_run:
    workflows: ["Build Search Indexes"]
    types: [completed]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Download index
        uses: actions/download-artifact@v3
        with:
          name: search-index

      - name: Build Docker image
        run: |
          docker build \
            -f Dockerfile.queryonly \
            -t myagent:latest .

      - name: Push to registry
        run: docker push myagent:latest
```

**Dockerfile.queryonly:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install query-only dependencies
RUN pip install signalwire-agents[search-queryonly]

# Copy pre-built index
COPY knowledge.swsearch /app/

# Copy agent code
COPY agent.py /app/

CMD ["python", "agent.py"]
```

## pgvector Strategy

With pgvector, the workflow is even simpler:

**Build once (CI/CD):**
```bash
sw-search ./docs \
  --backend pgvector \
  --connection-string "$DATABASE_URL" \
  --collection-name docs \
  --model mini
```

**Deploy everywhere (query-only):**
```bash
pip install signalwire-agents[search-queryonly]
# No index to copy - embeddings already in PostgreSQL
# Agents connect and query
```

All agents share the same pgvector database. No need to distribute .swsearch files.

## Hybrid Development Setup

Developers can use query-only locally by connecting to shared indexes:

**Team setup:**

1. **Shared development database:**
```bash
# DBA builds indexes to dev database
sw-search ./docs \
  --backend pgvector \
  --connection-string "postgresql://dev-db/knowledge" \
  --collection-name docs
```

2. **Developers install query-only:**
```bash
# Developers only need query-only
pip install signalwire-agents[search-queryonly]
```

3. **Agents connect to shared database:**
```python
agent.add_skill("native_vector_search", {
    "backend": "pgvector",
    "connection_string": os.getenv("DEV_DATABASE_URL"),
    "collection_name": "docs"
})
```

Benefits:
- Developers don't need full ML stack
- Faster local setup
- Consistent indexes across team
- Centralized index management

## Limitations of Query-Only

You **cannot** do these operations with query-only:

- ❌ **Build new indexes:**
```bash
sw-search ./docs --output new.swsearch
# Error: sentence-transformers not installed
```

- ❌ **Generate embeddings:**
```python
from signalwire_agents.search import generate_embedding
embedding = generate_embedding("text")
# Error: Model not available
```

- ❌ **Update indexes:**
```bash
sw-search ./new_docs --append knowledge.swsearch
# Error: Cannot generate embeddings
```

- ❌ **Migrate indexes:**
```bash
sw-search migrate ./old.swsearch --output new.swsearch
# Error: Requires embedding models
```

**Solution:** Use full search installation where you build indexes, query-only everywhere else.

## Checking Your Installation

Verify what's installed:

```python
from signalwire_agents.search import check_dependencies

check_dependencies()
# Output:
# ✓ SQLite backend available
# ✓ pgvector client available
# ✓ Vector operations available
# ✗ sentence-transformers not installed (query-only mode)
# ✗ Document processing not available (query-only mode)
```

## Real-World Example

**Company setup:**

**Build server (full search):**

- AWS EC2 instance
- `pip install signalwire-agents[search-full]`
- Nightly cron job builds indexes
- Uploads to S3 and pgvector

**Production agents (query-only):**

- 20 Kubernetes pods
- `pip install signalwire-agents[search-queryonly]`
- Download .swsearch from S3 on startup
- Or connect to pgvector directly

**Resource savings:**

- 20 pods × 400MB = 8GB saved
- Faster pod startup (smaller images)
- Less network bandwidth pulling images
- Lower storage costs

## Migration Path

If you're already using full search in production:

**1. Current state:**
```dockerfile
FROM python:3.11
RUN pip install signalwire-agents[search]
COPY . /app
```

**2. Separate index building:**
```bash
# Build index in CI/CD
sw-search ./docs --output knowledge.swsearch

# Store in artifact repository (S3, Artifactory, etc.)
aws s3 cp knowledge.swsearch s3://bucket/indexes/
```

**3. Update Dockerfile:**
```dockerfile
FROM python:3.11
RUN pip install signalwire-agents[search-queryonly]
COPY . /app

# Download pre-built index
RUN wget https://bucket/indexes/knowledge.swsearch -O /app/knowledge.swsearch
```

**4. Deploy:**
```bash
docker build -t agent:queryonly .
docker push agent:queryonly
kubectl rollout restart deployment/agent
```

Immediate savings in image size and deployment time.

## Key Takeaways

1. **Query-only saves ~100MB+** - smaller deployments
2. **Separate building from querying** - different environments
3. **Build indexes in CI/CD** - production doesn't need ML
4. **Perfect for production** - agents only query
5. **Works with both backends** - .swsearch files or pgvector
6. **Container-friendly** - smaller images, faster deployments
7. **Serverless-compatible** - fits within size limits

Next, we'll dive into the `sw-search` CLI tool in detail, exploring all the commands and workflows for building, validating, and managing search indexes.
# 11. Building the Index: CLI Deep Dive

The `sw-search` command-line tool is your interface to the search system. It builds indexes, validates them, searches them, and migrates between backends. Let's explore every command and workflow.

## Command Overview

The `sw-search` CLI has several subcommands:

```bash
sw-search [options] <paths>           # Build index (default)
sw-search validate <index>            # Validate index integrity
sw-search search <index> <query>      # Search within index
sw-search remote <url> <query>        # Search via remote API
sw-search migrate <source>            # Migrate between backends
```

## Building Indexes (Default Command)

The most common operation is building an index:

```bash
sw-search ./docs --output knowledge.swsearch
```

### Basic Usage

**Single directory:**
```bash
sw-search ./docs
# Output: docs.swsearch (default name)
```

**Multiple directories:**
```bash
sw-search ./docs ./examples ./tutorials
# Output: docs.swsearch
```

**Specific files:**
```bash
sw-search README.md CONTRIBUTING.md docs/api.md
```

**Mixed (directories and files):**
```bash
sw-search ./docs README.md ./examples/agent.py
```

### Output Options

**Specify output filename:**
```bash
sw-search ./docs --output knowledge.swsearch
```

**Output to different directory:**
```bash
sw-search ./docs --output /var/indexes/knowledge.swsearch
```

**Build to pgvector:**
```bash
sw-search ./docs \
  --backend pgvector \
  --connection-string "postgresql://user:pass@localhost/db" \
  --output docs_collection
```

Note: With pgvector, `--output` is the collection name, not a filename.

### File Type Filtering

**Include specific file types:**
```bash
sw-search ./docs --file-types md,txt,rst
```

**For code documentation:**
```bash
sw-search ./src --file-types py,js,ts,md
```

**PDF documents:**
```bash
sw-search ./manuals --file-types pdf
```

Supported file types:
- Text: `txt`, `md`, `rst`, `asciidoc`
- Code: `py`, `js`, `ts`, `java`, `cpp`, `c`, `go`, `rs`
- Documents: `pdf`, `docx`, `xlsx`, `pptx`
- Web: `html`, `xml`
- Data: `json`, `yaml`, `csv`

### Excluding Files

**Exclude patterns:**
```bash
sw-search ./docs \
  --exclude "**/test/**,**/__pycache__/**,**/node_modules/**"
```

**Common exclusions:**
```bash
sw-search ./project \
  --exclude "*.log,*.tmp,**/build/**,**/dist/**"
```

Patterns use glob syntax (`*`, `**`, `?`).

### Chunking Strategy Options

**Sentence-based (default):**
```bash
sw-search ./docs \
  --chunking-strategy sentence \
  --max-sentences-per-chunk 5 \
  --split-newlines 2
```

**Markdown-aware:**
```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --file-types md
```

**Sliding window:**
```bash
sw-search ./docs \
  --chunking-strategy sliding \
  --chunk-size 100 \
  --overlap-size 20
```

**Semantic:**
```bash
sw-search ./docs \
  --chunking-strategy semantic \
  --semantic-threshold 0.6
```

**QA-optimized:**
```bash
sw-search ./docs \
  --chunking-strategy qa
```

**JSON (pre-chunked):**
```bash
sw-search ./chunks.json \
  --chunking-strategy json \
  --file-types json
```

### Model Selection

**Use mini model (recommended):**
```bash
sw-search ./docs --model mini
```

**Use base model:**
```bash
sw-search ./docs --model base
```

**Use custom model:**
```bash
sw-search ./docs --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Metadata and Tags

**Add custom tags:**
```bash
sw-search ./docs --tags documentation,api,v2
```

**Tags are searchable:**
```python
agent.add_skill("native_vector_search", {
    "index_path": "./docs.swsearch",
    "tags": ["api"]  # Only search chunks tagged "api"
})
```

**Add language information:**
```bash
sw-search ./docs --languages en,es,fr
```

### Verbose Output

**See detailed progress:**
```bash
sw-search ./docs --verbose
```

Output:
```
Scanning files...
✓ Found 42 files (15 md, 12 py, 15 txt)

Processing files...
✓ docs/getting-started.md (3 chunks)
✓ docs/api-reference.md (12 chunks)
✓ examples/agent.py (5 chunks)
...

Generating embeddings...
[████████████████████] 100% (150/150 chunks)

Building index...
✓ Created vector index
✓ Created metadata tables
✓ Created text search index

Complete!
✓ Index saved to docs.swsearch
  Total chunks: 150
  Index size: 2.3 MB
  Avg chunk size: 245 words
  Model: mini (384 dims)
```

## Exporting to JSON

**Export all chunks to single file:**
```bash
sw-search ./docs \
  --output-format json \
  --output all_chunks.json
```

**Export to separate files:**
```bash
sw-search ./docs \
  --output-format json \
  --output-dir ./chunks/
```

Creates one JSON file per source document:
```
chunks/
  docs_getting-started.json
  docs_api-reference.json
  examples_agent.json
```

## Validating Indexes

Check index integrity:

```bash
sw-search validate ./knowledge.swsearch
```

Output:
```
Validating index...
✓ File format valid
✓ Metadata complete
✓ Embeddings present (150 chunks)
✓ Text content present
✓ Tags indexed
✓ Vector dimensions consistent (384)

Index statistics:
  Total chunks: 150
  Avg embedding norm: 1.00
  Text coverage: 100%
  Model: sentence-transformers/all-MiniLM-L6-v2
  Created: 2025-01-15 10:30:00

✓ Index is valid
```

**Validate pgvector collection:**
```bash
sw-search validate \
  --backend pgvector \
  --connection-string "postgresql://localhost/db" \
  --collection-name docs
```

## Searching Indexes

Test searches directly from CLI:

**Basic search:**
```bash
sw-search search ./knowledge.swsearch "how to create an agent"
```

Output:
```
Results for: "how to create an agent"

1. [Score: 0.87] docs/getting-started.md
   Creating Your First Agent

   To create an agent, inherit from AgentBase and define your configuration...

2. [Score: 0.82] docs/api-reference.md
   AgentBase Class

   The AgentBase class is the foundation for all agents...

3. [Score: 0.78] examples/simple_agent.py
   class MyAgent(AgentBase):
       def __init__(self):
           super().__init__(name="MyAgent")...
```

**Adjust result count:**
```bash
sw-search search ./knowledge.swsearch "query" --count 10
```

**Filter by tags:**
```bash
sw-search search ./knowledge.swsearch "query" --tags api,reference
```

**JSON output:**
```bash
sw-search search ./knowledge.swsearch "query" --json
```

Output:
```json
{
  "query": "how to create an agent",
  "results": [
    {
      "score": 0.87,
      "content": "To create an agent...",
      "metadata": {
        "filename": "getting-started.md",
        "section": "Creating Your First Agent"
      }
    }
  ]
}
```

**Verbose output (with scores):**
```bash
sw-search search ./knowledge.swsearch "query" --verbose
```

Shows detailed scoring breakdown:
```
Result 1:
  Vector score: 0.82
  Keyword matches: ["agent", "create"]
  Keyword boost: +0.15 (15%)
  Metadata boost: +0.10 (10%)
  Final score: 0.87

  Content: To create an agent...
```

## Remote Search

Query a remote search server:

**Basic remote search:**
```bash
sw-search remote http://localhost:8001 "query" --index-name docs
```

**With authentication:**
```bash
sw-search remote http://localhost:8001 "query" \
  --index-name docs \
  --auth-user admin \
  --auth-pass secret
```

**Specify result count:**
```bash
sw-search remote http://localhost:8001 "query" \
  --index-name docs \
  --count 10
```

Remote search is useful for:
- Centralized search service
- Multiple agents querying same index
- Separating search from agent infrastructure

## Migration Commands

Move indexes between backends:

**SQLite to pgvector:**
```bash
sw-search migrate ./knowledge.swsearch \
  --to-pgvector \
  --connection-string "postgresql://localhost/db" \
  --collection-name docs
```

**pgvector to SQLite:**
```bash
sw-search migrate \
  --from-pgvector \
  --connection-string "postgresql://localhost/db" \
  --collection-name docs \
  --output exported.swsearch
```

**Get migration info:**
```bash
sw-search migrate --info ./knowledge.swsearch
```

Output:
```
Index Information:
  Type: SQLite (.swsearch)
  Chunks: 150
  Model: mini (384 dims)
  Size: 2.3 MB
  Created: 2025-01-15

Migration options:
  ✓ Can migrate to pgvector
  ✓ Can export to JSON
  ✓ Can validate
```

## Environment Variables

Configure defaults via environment:

**Model selection:**
```bash
export SW_SEARCH_MODEL=mini
sw-search ./docs  # Uses mini model
```

**Default output:**
```bash
export SW_SEARCH_OUTPUT=/var/indexes/
sw-search ./docs  # Outputs to /var/indexes/docs.swsearch
```

**pgvector connection:**
```bash
export PGVECTOR_CONNECTION="postgresql://user:pass@localhost/db"
sw-search ./docs --backend pgvector --output docs
```

**Verbosity:**
```bash
export SW_SEARCH_VERBOSE=1
sw-search ./docs  # Verbose output by default
```

## Common Workflows

### Workflow 1: Development Iteration

```bash
# Build initial index
sw-search ./docs --output dev.swsearch

# Test search
sw-search search dev.swsearch "test query"

# Not happy? Try different chunking
sw-search ./docs \
  --chunking-strategy markdown \
  --output dev_markdown.swsearch

# Compare results
sw-search search dev.swsearch "test query"
sw-search search dev_markdown.swsearch "test query"

# Pick the better one
mv dev_markdown.swsearch knowledge.swsearch
```

### Workflow 2: JSON Curation

```bash
# Export to JSON
sw-search ./docs \
  --chunking-strategy markdown \
  --output-format json \
  --output chunks.json

# Edit chunks.json manually
vim chunks.json

# Build final index from curated JSON
sw-search chunks.json \
  --chunking-strategy json \
  --file-types json \
  --output curated.swsearch

# Validate
sw-search validate curated.swsearch
```

### Workflow 3: Production Deployment

```bash
# Build on build server
sw-search ./docs \
  --model mini \
  --chunking-strategy markdown \
  --output production.swsearch \
  --verbose

# Validate before deploying
sw-search validate production.swsearch

# Upload to S3 or artifact storage
aws s3 cp production.swsearch s3://bucket/indexes/

# Or push to pgvector
sw-search ./docs \
  --backend pgvector \
  --connection-string "$PROD_DATABASE_URL" \
  --collection-name production_docs \
  --model mini
```

### Workflow 4: Multi-Collection pgvector

```bash
# Build multiple collections
sw-search ./docs/api \
  --backend pgvector \
  --connection-string "$DATABASE_URL" \
  --collection-name api_docs

sw-search ./docs/guides \
  --backend pgvector \
  --connection-string "$DATABASE_URL" \
  --collection-name guides

sw-search ./docs/examples \
  --backend pgvector \
  --connection-string "$DATABASE_URL" \
  --collection-name examples

# Validate each
for collection in api_docs guides examples; do
  sw-search validate \
    --backend pgvector \
    --connection-string "$DATABASE_URL" \
    --collection-name $collection
done
```

## Batch Processing

**Process multiple directories:**
```bash
for dir in docs-v1 docs-v2 docs-v3; do
  sw-search ./$dir --output ${dir}.swsearch
done
```

**Build indexes in parallel:**
```bash
sw-search ./docs1 --output docs1.swsearch &
sw-search ./docs2 --output docs2.swsearch &
sw-search ./docs3 --output docs3.swsearch &
wait
```

## Debugging Build Issues

**Enable maximum verbosity:**
```bash
sw-search ./docs --verbose --debug
```

**Check file discovery:**
```bash
sw-search ./docs --verbose --dry-run
# Lists files that would be processed without building
```

**Test on single file:**
```bash
sw-search ./docs/problematic.md --output test.swsearch --verbose
```

**Validate after building:**
```bash
sw-search ./docs --output test.swsearch && \
sw-search validate test.swsearch
```

## Performance Tuning

**Batch size for embedding generation:**
```bash
sw-search ./docs --batch-size 100
# Default: 32, Lower = less memory, Higher = faster
```

**Parallel processing:**
```bash
sw-search ./docs --workers 4
# Use multiple CPU cores for processing
```

**Memory-constrained environments:**
```bash
sw-search ./docs \
  --batch-size 16 \
  --workers 1 \
  --model mini
```

## Key Takeaways

1. **sw-search is the primary CLI tool** - build, search, validate, migrate
2. **Default command builds indexes** - specify paths and options
3. **Many chunking strategies** - choose based on content type
4. **Validate before deploying** - catch issues early
5. **Test searches from CLI** - verify quality before using in agents
6. **Export to JSON for curation** - manual review and editing
7. **Environment variables for defaults** - configure once, use everywhere
8. **pgvector and SQLite use same CLI** - just different --backend flag

Next, we'll explore how to use search in your agents, including configuration options, custom response formatting, and handling voice vs chat modes differently.
# 12. Using Search in Your Agent

Now that you understand how to build indexes, let's explore how to integrate search into your agents. The `native_vector_search` skill makes it simple, but there are many configuration options to optimize for your use case.

## Basic Integration

The simplest integration uses default settings:

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

        # Agent will automatically use this function when needed
```

That's it. The agent now has access to your knowledge base.

## Configuration Options Explained

The `native_vector_search` skill has many configuration options:

### Required Parameters

**tool_name** (string)
- Name of the SWAIG function exposed to the LLM
- Should be descriptive: `search_docs`, `search_api`, `search_knowledge`

**description** (string)
- Tells the LLM when to use this function
- Be specific: "Search the API documentation for information about endpoints, authentication, and request formats"

### Backend Selection (Choose One)

**Option 1: Local .swsearch file**
```python
{
    "index_path": "./knowledge.swsearch"
}
```

**Option 2: Remote search server**
```python
{
    "remote_url": "http://localhost:8001",
    "index_name": "docs"
}
```

**Option 3: pgvector database**
```python
{
    "backend": "pgvector",
    "connection_string": "postgresql://user:pass@localhost:5432/db",
    "collection_name": "docs",
    "model_name": "mini"  # Must match model used during indexing
}
```

### Search Behavior Parameters

**count** (integer, default: 5)
- Number of results to return
- More results = more context, but more tokens used

```python
{
    "count": 3  # Return top 3 results
}
```

**distance_threshold** (float, default: 0.5)
- Minimum similarity score (0.0 to 1.0)
- Lower = stricter matching
- Higher = more permissive

```python
{
    "distance_threshold": 0.4  # Only results with similarity > 0.4
}
```

**tags** (list, optional)
- Filter results by tags
- Only chunks with these tags will be returned

```python
{
    "tags": ["api", "reference"]  # Only API reference chunks
}
```

### User Experience Parameters

**no_results_message** (string, optional)
- Message when no results are found
- Use `{query}` placeholder for the search query

```python
{
    "no_results_message": "I couldn't find information about '{query}' in the documentation. Try rephrasing your question."
}
```

**max_content_length** (integer, default: 32768)
- Maximum total characters in response
- Prevents response truncation
- Budget is distributed across results

```python
{
    "max_content_length": 16384  # 16KB total
}
```

**swaig_fields** (dict, optional)
- SWAIG-specific configuration
- Includes function fillers for better UX

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

## Complete Configuration Example

Here's a fully configured search skill:

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

## Multiple Search Skills

An agent can have multiple search skills for different knowledge domains:

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
            "tags": ["api"],  # Only API-tagged chunks
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

The LLM will choose which search function to use based on the question.

## Custom Response Formatting

You can customize how search results are formatted before being sent to the LLM:

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
        formatted = "📚 Documentation Search Results:\n\n"
        formatted += f"Query: {query}\n"
        formatted += f"Found {len(results)} relevant sections:\n\n"

        # Add results
        formatted += response

        # Add footer
        formatted += "\n\n💡 Based on these results, provide a clear and accurate answer."

        return formatted
```

The callback receives:
- `response`: Default formatted response
- `agent`: The agent instance
- `query`: The search query
- `results`: List of result dictionaries
- `**kwargs`: Additional metadata

## Voice vs Chat Mode Considerations

For agents that handle both voice and chat, you might want different formatting:

```python
class MultiModalAgent(AgentBase):
    def __init__(self):
        super().__init__(name="MultiModalAgent")

        # Store whether this is voice or chat
        self.is_voice = False  # Will be set dynamically

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

        # Check if this is a voice interaction
        is_voice = getattr(agent, 'is_voice', False)

        if is_voice:
            # Voice mode - conversational instructions
            instructions = (
                "📞 Voice Mode:\n"
                "Use these search results to answer naturally. "
                "Don't read URLs or code verbatim. "
                "Summarize technical concepts clearly. "
                "Mention that detailed documentation is available online.\n\n"
            )
        else:
            # Chat mode - include links and code
            instructions = (
                "💬 Chat Mode:\n"
                "Use these search results to answer. "
                "Include relevant URLs from results. "
                "Format code with markdown code blocks. "
                "Provide comprehensive technical details.\n\n"
            )

        return instructions + response
```

## Real-World Example: Sigmond's Search Configuration

Sigmond uses multiple search skills with custom formatting:

```python
class SigmondAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Sigmond")

        # Build pgvector connection string
        pg_user = os.getenv("PGVECTOR_DB_USER", "signalwire")
        pg_pass = os.getenv("PGVECTOR_DB_PASSWORD")
        pg_host = os.getenv("PGVECTOR_HOST", "localhost")
        pg_port = os.getenv("PGVECTOR_PORT", "5432")
        pg_db = os.getenv("PGVECTOR_DB_NAME", "knowledge")

        connection_string = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        # SignalWire unified documentation
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

        # Pricing information
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

    def _format_search_results(self, response, agent, query, results, **kwargs):
        """Custom formatter that adapts to voice vs chat"""
        if not results:
            return response

        is_voice = getattr(agent, 'is_voice', False)

        if is_voice:
            instructions = (
                "📞 **Voice Mode Instructions:**\n"
                "- Provide natural, conversational responses\n"
                "- Don't read URLs or code snippets verbatim\n"
                "- Summarize technical concepts clearly\n"
                "- Mention code examples are in developer docs\n\n"
            )
        else:
            instructions = (
                "💬 **Chat Mode Instructions:**\n"
                "- Include relevant URLs from results\n"
                "- Format code with markdown blocks\n"
                "- Provide comprehensive details\n"
                "- Use markdown formatting\n\n"
            )

        return instructions + response
```

## Prompt Engineering for Search

Tell your agent how and when to use search:

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

## Testing Search Integration

Use `swaig-test` to verify search works:

```bash
# List available tools
swaig-test agent.py --list-tools

# Test search function
swaig-test agent.py --exec search_docs --query "how to create an agent"
```

You'll see the raw search results returned by the function.

## Debugging Search Issues

**Enable verbose logging:**

```python
import os
os.environ['SIGNALWIRE_LOG_LEVEL'] = 'DEBUG'

agent = MyAgent()
agent.run()
```

**Check if search is being called:**

- Monitor agent logs for search function calls
- Look for "Executing function: search_docs"
- Review search queries and results

**Test search directly:**
```bash
# Verify index works
sw-search search ./knowledge.swsearch "test query"
```

**Common issues:**

1. **Agent doesn't use search:**
   - Check function description is clear
   - Add prompt instructions to use search
   - Verify LLM understands when to search

2. **No results returned:**
   - Check distance_threshold (might be too strict)
   - Verify query matches content semantically
   - Test same query with `sw-search` CLI

3. **Poor quality results:**
   - Review chunking strategy
   - Check if content is well-structured
   - Consider markdown strategy for code docs

4. **Response truncated:**
   - Increase max_content_length
   - Reduce count (fewer results)
   - Check for very long chunks

## Performance Monitoring

Track search performance:

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

## Key Takeaways

1. **native_vector_search skill is simple to use** - just add to agent
2. **Many configuration options** - tune for your use case
3. **Multiple search skills supported** - different knowledge domains
4. **Custom formatters powerful** - adapt output to voice/chat
5. **Prompt engineering important** - tell agent when to search
6. **Test with swaig-test** - verify before deployment
7. **Monitor performance** - track search quality and speed

Next, we'll explore advanced topics like metadata filtering, tag-based searching, and building specialized search tools for specific use cases.
# 13. Advanced Topics: Metadata and Tags

Metadata and tags transform basic vector search into a sophisticated knowledge retrieval system. They add structure, enable filtering, and boost relevance through hybrid search. Let's explore advanced metadata techniques.

## Understanding Metadata

Every chunk has metadata - information about the chunk beyond its content:

```python
{
    "content": "The AgentBase class is the foundation...",
    "metadata": {
        "filename": "agent_base.py",
        "section": "Core Classes",
        "h1": "API Reference",
        "h2": "AgentBase",
        "tags": ["api", "reference", "core"],
        "has_code": True,
        "difficulty": "intermediate",
        "category": "development"
    }
}
```

This metadata enables:
- **Filtering**: Only search specific categories
- **Boosting**: Prioritize chunks with certain metadata
- **Organization**: Structure your knowledge base
- **Context**: Understand where information came from

## Automatic Metadata

The SDK adds metadata automatically based on chunking strategy:

### Sentence Strategy Metadata

```python
{
    "chunk_method": "sentence",
    "chunk_index": 5,
    "sentence_count": 8,
    "filename": "docs.md"
}
```

### Markdown Strategy Metadata

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

### QA Strategy Metadata

```python
{
    "chunk_method": "qa_optimized",
    "chunk_index": 2,
    "has_question": True,
    "has_process": True,
    "sentence_count": 6
}
```

## Adding Custom Metadata

### During Indexing (CLI)

Add tags when building the index:

```bash
sw-search ./docs \
  --tags documentation,api,v2,production \
  --output docs.swsearch
```

All chunks get these tags.

### In JSON Workflow

Add any metadata you want:

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

### Programmatic Metadata

Generate metadata when creating JSON:

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

## Tag-Based Filtering

Tags enable precise filtering during search:

### In Agent Configuration

```python
# Only search API documentation
self.add_skill("native_vector_search", {
    "tool_name": "search_api",
    "description": "Search API documentation",
    "index_path": "./docs.swsearch",
    "tags": ["api", "reference"]  # Only these tags
})
```

### Multiple Search Skills by Tag

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

The LLM will choose the appropriate search based on the question's complexity.

### Dynamic Tag Filtering

Pass tags at query time:

```python
def search_with_tags(self, query, tags):
    """Search with dynamic tag filtering"""
    # This would require custom SWAIG function
    # that accepts tags as parameters
    pass
```

## Metadata Boosting in Hybrid Search

Remember hybrid search? Metadata matching provides confirmation signals:

**Scenario:** User searches for "python authentication example"

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
- Metadata matches: "python", "authentication", "example" (3 matches)
- Boost: +30%
- Has "code" tag and keywords matched: +20%
- **Final score: 0.75 × 1.30 × 1.20 = 1.17**

**Chunk B:**
```python
{
    "content": "Authentication is important for security...",
    "metadata": {
        "tags": ["security", "authentication"]
    }
}
```
- Vector similarity: 0.82 (higher!)
- Metadata matches: "authentication" (1 match)
- Boost: +15%
- **Final score: 0.82 × 1.15 = 0.94**

**Result:** Chunk A ranks higher despite lower vector similarity because metadata confirmed it's exactly what the user wants.

## Organizing Knowledge by Category

Use categories to organize large knowledge bases:

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

Then create category-specific search:

```python
# Troubleshooting only
self.add_skill("native_vector_search", {
    "tool_name": "search_troubleshooting",
    "description": "Search troubleshooting guides for error solutions",
    "index_path": "./docs.swsearch",
    "tags": ["troubleshooting", "errors"]
})
```

## Priority-Based Metadata

Mark important content:

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

Use in custom formatter to highlight:

```python
def _format_with_priority(self, response, agent, query, results, **kwargs):
    """Highlight high-priority results"""
    formatted = ""

    for result in results:
        priority = result.get('metadata', {}).get('priority', 'normal')

        if priority == 'critical':
            formatted += "🚨 CRITICAL: "
        elif priority == 'high':
            formatted += "⚠️  HIGH PRIORITY: "

        formatted += result['content'] + "\n\n"

    return formatted
```

## Temporal Metadata

Track when content was created/updated:

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

Filter for recent content:

```python
# Get recent docs (in custom implementation)
results = search_with_filter(
    query="async support",
    metadata_filter={"version": "2.0"}
)
```

## Relationship Metadata

Link related chunks:

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

Use to build learning paths:

```python
def get_learning_path(self, start_chunk_id):
    """Build a learning path from metadata"""
    path = [start_chunk_id]
    current = self.get_chunk(start_chunk_id)

    while "next_topic" in current.get("metadata", {}):
        next_id = current["metadata"]["next_topic"]
        path.append(next_id)
        current = self.get_chunk(next_id)

    return path
```

## Language Metadata

For multilingual content:

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

Filter by language:

```python
self.add_skill("native_vector_search", {
    "tool_name": "search_french_docs",
    "description": "Rechercher la documentation en français",
    "index_path": "./docs.swsearch",
    "tags": ["french"]
})
```

## Audience Metadata

Tailor content to audience:

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

## Building Specialized Search Tools

Combine metadata with custom logic:

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
        """Format code examples with syntax highlighting hints"""
        formatted = "💻 Code Examples Found:\n\n"

        for result in results:
            languages = result.get('metadata', {}).get('code_languages', [])
            if languages:
                formatted += f"Languages: {', '.join(languages)}\n"

        formatted += response
        return formatted
```

## Metadata Best Practices

### 1. Be Consistent

Use the same metadata fields across your knowledge base:

**Good:**
```json
{"category": "api", "difficulty": "beginner"}
{"category": "troubleshooting", "difficulty": "intermediate"}
```

**Bad:**
```json
{"category": "api", "difficulty": "beginner"}
{"type": "troubleshooting", "level": "medium"}
```

### 2. Use Tags Liberally

More tags = more opportunities for matching:

```json
{
  "tags": [
    "authentication",
    "security",
    "api",
    "bearer-token",
    "auth",
    "login",
    "credentials"
  ]
}
```

### 3. Include Synonyms

Users might search with different terms:

```json
{
  "tags": [
    "installation",
    "setup",
    "getting-started",
    "install",
    "configure"
  ]
}
```

### 4. Structure Hierarchically

```json
{
  "category": "development",
  "subcategory": "testing",
  "topic": "unit-tests"
}
```

### 5. Track Metadata Provenance

```json
{
  "source": "official-docs",
  "author": "signalwire-team",
  "verified": true,
  "last_reviewed": "2025-01-15"
}
```

## Querying Metadata Directly

Some use cases need direct metadata queries:

```python
# Find all chunks about a topic
chunks = index.query_by_metadata(
    tags=["authentication", "security"]
)

# Find chunks by category
chunks = index.query_by_metadata(
    category="troubleshooting"
)

# Complex filters
chunks = index.query_by_metadata(
    difficulty="beginner",
    category="getting-started",
    has_code=True
)
```

This is useful for:
- Building table of contents
- Listing available topics
- Generating documentation indexes
- Analytics and reporting

## Key Takeaways

1. **Metadata adds structure** - organize beyond text content
2. **Tags enable filtering** - precise search by category
3. **Metadata boosts hybrid search** - confirmation signals
4. **Automatic metadata from chunking** - headers, code, structure
5. **Custom metadata in JSON** - add any fields you need
6. **Multiple search tools by tag** - specialized functions
7. **Consistent metadata schema** - makes filtering predictable

Next, we'll explore search quality tuning: adjusting parameters like distance_threshold and result count to optimize for your specific content and use case.
# 14. Search Quality: Tuning Your Results

Building an index is just the first step. To get the best search results, you need to tune parameters based on your content and use case. Let's explore how to optimize search quality.

## The Key Parameters

Three parameters have the biggest impact on search quality:

1. **distance_threshold** - How strict matching is
2. **count** - How many results to return
3. **max_content_length** - Total response size budget

Let's understand each one.

## Distance Threshold: Strictness Control

The `distance_threshold` parameter (0.0 to 1.0) controls how similar results must be to the query.

### How It Works

Vector similarity is measured as cosine similarity:
- 1.0 = identical vectors (perfect match)
- 0.8 = very similar
- 0.5 = somewhat similar
- 0.2 = barely related
- 0.0 = completely different

The threshold filters out low-similarity results:

```python
{
    "distance_threshold": 0.4  # Only return results with similarity > 0.4
}
```

### Finding the Right Threshold

**Too strict (0.7+):**
```
Query: "how to configure voice settings"
Threshold: 0.7

Results: 0 results found
Problem: No chunks meet the high threshold
```

**Too permissive (0.2-):**
```
Query: "how to configure voice settings"
Threshold: 0.2

Results: 15 results (many irrelevant)
- "The AgentBase class provides configuration..." (0.32)
- "Voice parameters can be adjusted..." (0.78)
- "Setting up database connections..." (0.23)
- "Debugging your application..." (0.25)
Problem: Too much noise, low-quality matches included
```

**Just right (0.4-0.5):**
```
Query: "how to configure voice settings"
Threshold: 0.4

Results: 3 high-quality results
- "Voice parameters can be adjusted..." (0.78)
- "Configuring voice with add_language()..." (0.65)
- "Voice settings reference..." (0.52)
Success: Only relevant results
```

### Recommendations by Content Type

**Technical documentation (code, APIs):**
```python
{
    "distance_threshold": 0.4  # Balanced
}
```
Technical docs are specific. Higher threshold prevents off-topic matches.

**General knowledge base (FAQs, guides):**
```python
{
    "distance_threshold": 0.5  # Moderate
}
```
More conversational content benefits from broader matching.

**Creative content (blogs, articles):**
```python
{
    "distance_threshold": 0.6  # Permissive
}
```
Creative content has more varied language, needs wider net.

**Precise lookups (error codes, model names):**
```python
{
    "distance_threshold": 0.3  # Strict
}
```
When exact matches matter, be strict.

### Testing Threshold Values

Use the CLI to experiment:

```bash
# Try different thresholds
sw-search search ./docs.swsearch "your query" --threshold 0.3 --verbose
sw-search search ./docs.swsearch "your query" --threshold 0.4 --verbose
sw-search search ./docs.swsearch "your query" --threshold 0.5 --verbose
```

Look at the similarity scores in verbose output:
```
Result 1: similarity=0.82
Result 2: similarity=0.67
Result 3: similarity=0.45
Result 4: similarity=0.38
Result 5: similarity=0.22
```

If threshold is 0.4, results 4 and 5 are excluded. Are they relevant? Adjust threshold accordingly.

### Dynamic Threshold Strategy

Some agents adjust threshold based on result count:

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

## Result Count: Balancing Context and Noise

The `count` parameter determines how many results to return:

```python
{
    "count": 5  # Return top 5 results
}
```

### Trade-offs

**Fewer results (1-3):**

- ✅ Precise answers
- ✅ Less noise
- ✅ Faster LLM processing
- ✅ Lower token costs
- ❌ Might miss context
- ❌ Limited perspective

**More results (5-10):**

- ✅ More context
- ✅ Multiple perspectives
- ✅ Better coverage
- ❌ More noise
- ❌ Slower processing
- ❌ Higher token costs

**Too many results (10+):**

- ❌ Information overload
- ❌ Diminishing returns
- ❌ Response truncation risk
- ❌ Expensive

### Recommendations by Query Type

**Specific questions:**
```python
{
    "count": 3  # Focused
}
```
"How do I authenticate?" → Need specific answer, not broad overview.

**Exploratory questions:**
```python
{
    "count": 5  # Moderate
}
```
"What are the authentication options?" → Need comprehensive view.

**Research queries:**
```python
{
    "count": 7  # Comprehensive
}
```
"Tell me everything about authentication" → Want thorough coverage.

### Result Quality Distribution

Remember hybrid search retrieves 3x the requested count, then ranks:

```python
{
    "count": 5  # Request 5 results
}
```

Internally:
1. Vector search retrieves 15 candidates
2. Hybrid scoring ranks all 15
3. Top 5 are returned

This ensures the 5 you get are the best possible from a larger pool.

## Content Length Budgeting

The `max_content_length` parameter controls total response size:

```python
{
    "max_content_length": 32768  # 32KB total (default)
}
```

### How Budget is Distributed

If you request 5 results with 32KB budget:

```python
overhead_per_result = 300 chars  # Metadata, formatting
total_overhead = 5 * 300 = 1500 chars
available_for_content = 32768 - 1500 = 31268 chars
per_result_limit = 31268 / 5 = 6253 chars per result
```

Each result gets ~6KB of content, keeping total under 32KB.

### Why This Matters

**LLM context limits:**

- Models have token limits (8K, 16K, 128K)
- Search results consume context
- Must leave room for conversation

**Response quality:**

- Too much content = information overload
- Too little = missing context
- Balance is key

### Tuning Content Length

**Voice agents (responses spoken aloud):**
```python
{
    "max_content_length": 16384  # 16KB
}
```
Less content = faster, more focused responses.

**Chat agents (text responses):**
```python
{
    "max_content_length": 32768  # 32KB (default)
}
```
Can handle more detailed information.

**Complex queries needing depth:**
```python
{
    "max_content_length": 65536  # 64KB
}
```
Comprehensive answers with lots of context.

### Monitoring Truncation

Check if results are being truncated:

```python
def _format_with_truncation_warning(self, response, agent, query, results, **kwargs):
    """Warn if results were truncated"""
    truncated = kwargs.get('truncated', False)

    if truncated:
        response += "\n\n⚠️ Some results were truncated due to length limits. For complete information, visit the documentation."

    return response
```

## Testing and Iteration

### Create Test Queries

Build a test suite of representative queries:

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
```

### Test Different Configurations

```python
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

### Evaluate Result Quality

For each configuration, evaluate:
1. **Precision**: Are results relevant?
2. **Recall**: Did we find all relevant content?
3. **Coverage**: Do results answer the question?
4. **Diversity**: Are results covering different aspects?

### Example Evaluation

**Query:** "python authentication examples"

**Config A: threshold=0.3, count=3**
```
Results:
1. Python auth example (relevant ✅)
2. Authentication overview (relevant ✅)
3. General Python guide (not specific ❌)

Precision: 67% (2/3 relevant)
```

**Config B: threshold=0.4, count=5**
```
Results:
1. Python auth example (relevant ✅)
2. Python bearer token code (relevant ✅)
3. Authentication overview (relevant ✅)
4. Python OAuth example (relevant ✅)
5. Auth troubleshooting (somewhat relevant ✓)

Precision: 90% (4.5/5 relevant)
```

Config B wins - better precision and more comprehensive.

## Debug Mode for Understanding Scores

Enable verbose logging to see scoring:

```python
import os
os.environ['SEARCH_DEBUG'] = '1'

results = agent.search("query")
```

Output:
```
Query: "python authentication examples"

Candidate pool: 15 chunks

Chunk 1:
  Vector score: 0.82
  Keyword matches: ["python", "authentication", "examples"]
  Keyword boost: +0.45 (3 matches × 0.15)
  Has 'code' tag: +0.20
  Final score: 0.82 × 1.45 × 1.20 = 1.43

Chunk 2:
  Vector score: 0.75
  Keyword matches: ["authentication"]
  Keyword boost: +0.15 (1 match × 0.15)
  Final score: 0.75 × 1.15 = 0.86

Returning top 5 results...
```

This reveals why certain results rank higher.

## Common Issues and Fixes

### Issue: No Results

**Symptoms:**
```
Query: "setting up authentication"
Results: []
```

**Diagnosis:**

- Threshold too strict
- Content doesn't cover this topic
- Query phrasing doesn't match content

**Fixes:**
1. Lower threshold: `0.5 → 0.4 → 0.3`
2. Test query variations: "authentication setup", "configuring auth"
3. Add content if truly missing

### Issue: Irrelevant Results

**Symptoms:**
```
Query: "Python API examples"
Results:
- General API overview (not Python-specific)
- Ruby examples
- Installation instructions
```

**Diagnosis:**

- Threshold too permissive
- Poor metadata tagging
- Chunking mixed unrelated content

**Fixes:**
1. Raise threshold: `0.4 → 0.5`
2. Add tags for filtering: `tags=["python", "code"]`
3. Improve chunking strategy (use markdown for code)

### Issue: Best Result Not First

**Symptoms:**
```
Query: "voice configuration"
Results:
1. General configuration overview (0.75)
2. Perfect voice config guide (0.82)  ← Should be #1
```

**Diagnosis:**

- Hybrid scoring not boosting correctly
- Missing metadata on best result

**Fixes:**
1. Add tags to best result: `["voice", "configuration"]`
2. Check metadata matches query terms
3. Verify code chunks have "code" tag

### Issue: Response Truncated

**Symptoms:**
```
Results look cut off mid-sentence...
```

**Diagnosis:**

- max_content_length too low
- Individual chunks too long

**Fixes:**
1. Increase budget: `32768 → 65536`
2. Reduce count: `7 → 5` (fewer results = more space each)
3. Improve chunking (shorter chunks)

## A/B Testing in Production

Track which configuration works best:

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

Monitor metrics:
- User satisfaction
- Follow-up questions (did first answer suffice?)
- Query success rate

## Key Takeaways

1. **distance_threshold controls strictness** - 0.4-0.5 is good default
2. **count balances context and noise** - 3-5 results for most queries
3. **max_content_length prevents truncation** - 32KB default, adjust as needed
4. **Test with real queries** - build test suite, iterate
5. **Debug mode reveals scoring** - understand why results rank
6. **Common issues have fixes** - no results, irrelevance, truncation
7. **A/B test in production** - find optimal configuration empirically

Next, we'll explore a real-world case study: how Sigmond, our production agent, uses search at scale with multiple collections, custom formatting, and production-grade configuration.
# 15. Real-World Example: Building Sigmond

Sigmond is SignalWire's production demo agent - a sophisticated AI assistant that showcases the platform's capabilities while answering real questions about SignalWire products, pricing, and technical documentation. Let's examine how Sigmond uses search at scale.

## The Challenge

Sigmond needs to answer questions about:
- **SignalWire platform** - SDKs, APIs, services, features
- **Pricing** - Plans, costs, billing details
- **FreeSWITCH** - Telephony system documentation
- **Technical implementation** - Code examples, configuration
- **Business information** - Use cases, competitive positioning

A single knowledge base wouldn't work well. Different query types need different content sources.

## The Architecture

Sigmond uses **three separate knowledge bases** in a single PostgreSQL database:

```
PostgreSQL (pgvector)
├── signalwire_unified (5,000+ chunks)
│   ├── SDK documentation
│   ├── Developer guides
│   ├── API references
│   └── Platform features
├── pricing (500+ chunks)
│   ├── Pricing pages
│   ├── Plan comparisons
│   └── Billing information
└── freeswitch (2,000+ chunks)
    ├── FreeSWITCH documentation
    ├── Telephony concepts
    └── SIP configuration
```

Each collection is optimized for its content type.

## Building the Collections

### 1. SignalWire Unified Documentation

Built from multiple sources with markdown strategy:

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

**Why markdown strategy:**

- SDK docs have lots of code examples
- API reference needs intact code blocks
- Header hierarchy provides context

**Result:**

- 5,000+ chunks
- Code examples properly tagged
- Hierarchical metadata preserved

### 2. Pricing Collection

Built with JSON strategy for precise pricing information:

```bash
# First, use AI to structure pricing data into perfect JSON format
# This allows exact control over how pricing is chunked and presented

sw-search \
  ./pricing.json \
  --chunking-strategy json \
  --model mini \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name pricing \
  --tags pricing,plans,costs
```

**Why JSON strategy:**

- Pricing needs precision - no auto-chunking errors
- AI can analyze pricing pages and format into optimal JSON
- Complete control over chunk boundaries
- Each plan, feature, or price point as its own chunk
- Consistent formatting across all pricing info

**Result:**

- 500+ chunks
- Each pricing detail perfectly chunked
- High relevance for pricing queries
- No risk of splitting prices from descriptions

### 3. FreeSWITCH Documentation

Built from technical documentation:

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

**Result:**

- 2,000+ chunks
- Telephony-specific content
- Separate from SignalWire docs (avoids confusion)

## Agent Configuration

Here's how Sigmond configures the three search skills:

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

        # Add search skills
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
```

## Custom Response Formatting

Sigmond formats results differently for voice vs chat:

```python
def _format_search_results(self, response, agent, query, results, **kwargs):
    """Custom formatter that adapts to voice vs chat mode"""
    if not results:
        return response  # Use default no_results_message

    # Check if this is voice or chat
    is_voice = getattr(agent, 'is_voice', False)

    if is_voice:
        # Voice mode - conversational instructions
        instructions = (
            "📞 **Voice Mode Instructions:**\n"
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
        # Chat mode - include links and code
        instructions = (
            "💬 **Chat Mode Instructions:**\n"
            "Use the following search results to answer the user's question. "
            "Since this is a text chat:\n"
            "- Include relevant URLs from the results in your response\n"
            "- Format all code examples with markdown code blocks (```language```)\n"
            "- Scrape any relevant URLs to get more detailed information\n"
            "- Provide comprehensive technical details when appropriate\n"
            "- Use markdown formatting for better readability\n"
            "- If there is not enough info in the response, try searching the web\n\n"
        )

    # Prepend instructions to the response
    return instructions + response
```

**Why this matters:**

**Voice mode:**

- "The add_language method configures voice settings. You pass parameters like name, code, and voice."
- Clean, speakable

**Chat mode:**

- "Use the `add_language()` method:\n```python\nagent.add_language(name='English', code='en-US', voice='elevenlabs.adam')\n```"
- Code included, formatted

## Prompt Engineering for Search

Sigmond's prompt instructs when to use each search:

```python
self.prompt_add_section(
    "Using Your Tools",
    body="Match the right tool to each question:",
    bullets=[
        "SignalWire technical/SDK/API questions → search_signalwire_knowledge",
        "Pricing/costs questions → search_pricing",
        "FreeSWITCH/telephony questions → search_freeswitch_knowledge",
        "Current events/general info → web_search",
        "Specific URLs → scrape_url or crawl_site",
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

This teaches the LLM:
- Which search to use for which question
- Always search before answering
- Specific tool for pricing

## Deployment Configuration

Sigmond runs in production with:

**Environment variables:**
```bash
export PGVECTOR_DB_USER=signalwire
export PGVECTOR_DB_PASSWORD=<secure-password>
export PGVECTOR_DB_NAME=sigmond_knowledge
export PGVECTOR_HOST=postgres.production.local
export PGVECTOR_PORT=5432
```

**Multiple instances:**
```
Kubernetes Deployment
├── sigmond-pod-1 ──┐
├── sigmond-pod-2 ──┼──> PostgreSQL (pgvector)
├── sigmond-pod-3 ──┘
└── sigmond-pod-4 ──┘
```

All pods share the same pgvector database. No index duplication.

**Search-queryonly installation:**
```dockerfile
FROM python:3.11-slim
RUN pip install signalwire-agents[search-queryonly]
COPY sigmond.py /app/
CMD ["python", "/app/sigmond.py"]
```

Pods don't need ML models - just query pre-built indexes.

## Performance Metrics

In production, Sigmond achieves:

**Query performance:**

- SignalWire docs (5k chunks): ~30ms
- Pricing (500 chunks): ~10ms
- FreeSWITCH (2k chunks): ~20ms

**Success rate:**

- 95% of technical questions answered from search
- 98% of pricing questions answered from search
- 5% fallback to web search for current info

**User satisfaction:**

- Voice users appreciate concise answers
- Chat users value code examples and links
- Both benefit from accurate, sourced information

## Lessons Learned

### 1. Multiple Collections Beat Single Index

**Initially tried:** One large index with all content

**Problem:** Queries would mix:
- SDK docs with pricing info
- FreeSWITCH with SignalWire concepts
- Results were confusing

**Solution:** Separate collections by domain
- LLM chooses right search
- Results are focused
- Better accuracy

### 2. Markdown Strategy Essential for Technical Docs

**Initially tried:** Sentence chunking

**Problem:**

- Code blocks split mid-code
- Lost context (which class is this method from?)
- Users couldn't find code examples

**Solution:** Markdown strategy
- Code blocks intact
- Header hierarchy preserved
- "code" tags boost code examples

### 3. Custom Formatting for Voice vs Chat

**Initially tried:** Same formatting for both

**Problem:**

- Voice read URLs aloud ("h-t-t-p-s colon slash slash...")
- Chat responses lacked links and code

**Solution:** Adaptive formatting
- Voice: Conversational, no URLs
- Chat: Links, formatted code blocks
- Much better UX

### 4. Prompt Engineering Matters

**Initially:** "You have search functions"

**Problem:** LLM didn't know when to use which search

**Solution:** Explicit instructions
- "For pricing → search_pricing"
- "For SDK → search_signalwire_knowledge"
- "ALWAYS search before answering"

Search usage went from 60% to 95% of queries.

### 5. Mini Model Sufficient

**Initially:** Used base model (768 dims)

**Problem:** Slower queries, larger index

**Testing:** Compared mini vs base
- Quality difference: ~2%
- Speed difference: 2x faster
- Size difference: 50% smaller

**Decision:** Use mini
- Negligible quality loss
- Significant performance gain

### 6. Distance Threshold Sweet Spot

**Tested:** 0.3, 0.4, 0.5, 0.6

**Results:**

- 0.3: Too strict, often zero results
- 0.4: Good balance ✅
- 0.5: Some irrelevant results
- 0.6: Too permissive, lots of noise

**Decision:** 0.4 for all collections

## Maintenance and Updates

### Updating Collections

Pricing changes frequently. Update workflow:

```bash
# Build updated pricing index
sw-search ./pricing-updated \
  --chunking-strategy qa \
  --model mini \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name pricing_v2

# Test with queries
sw-search search \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name pricing_v2 \
  "what does voice cost"

# Switch agents to new collection (rolling update)
kubectl set env deployment/sigmond \
  PRICING_COLLECTION=pricing_v2

# Delete old collection after verification
```

No downtime. Agents switch collections dynamically.

### Monitoring Search Health

Track metrics:
- Query count per collection
- Average search time
- No-result queries (indicates gaps)
- Fallback to web search rate

## Cost Analysis

**PostgreSQL hosting:**

- Database: 20GB (~$50/month)
- CPU: Moderate (vector ops)
- Total: ~$100/month

**vs Alternatives:**

- Pinecone: $70/month + per-query costs
- OpenAI Assistants: $0.20 per 1K assistant API calls
- Custom solution: Higher development cost

**Verdict:** pgvector is cost-effective at scale.

## Scalability

Current scale:
- 7,500+ total chunks
- 10+ concurrent agents
- 1,000+ queries/day

Easy to scale:
- Add read replicas for more throughput
- Partition collections for millions of chunks
- Connection pooling handles 100+ agents

## Key Takeaways from Sigmond

1. **Multiple collections organize knowledge** - domain-specific search
2. **pgvector scales for production** - concurrent access, shared index
3. **Markdown strategy for technical docs** - code examples critical
4. **Custom formatting by mode** - voice vs chat needs differ
5. **Prompt engineering drives usage** - explicit instructions needed
6. **Mini model performs well** - 2% quality loss, 2x speed gain
7. **0.4 threshold is sweet spot** - for technical documentation
8. **Monitoring reveals gaps** - track no-result queries
9. **Rolling updates work well** - new collections, zero downtime
10. **Cost-effective at scale** - cheaper than managed alternatives

Sigmond demonstrates that the search system scales from simple single-agent deployments to production multi-agent systems with thousands of chunks and hundreds of queries per day.

Next, we'll compare the SignalWire Agents SDK search approach to other popular RAG implementations and explain why integrated search provides advantages.
# 16. Comparison to Other Approaches

The SignalWire Agents SDK isn't the only way to add knowledge retrieval to AI agents. Let's compare it to popular alternatives and understand the trade-offs.

## The Landscape

Common RAG approaches:

1. **OpenAI Assistants API** - Hosted solution from OpenAI
2. **LangChain + External Vector DB** - Framework + managed database
3. **Embedding API + Custom Search** - DIY with embedding services
4. **SignalWire Agents SDK** - Integrated search

Let's compare them.

## OpenAI Assistants API

### How It Works

Upload documents to OpenAI, they handle everything:

```python
from openai import OpenAI
client = OpenAI()

# Upload files
file = client.files.create(
    file=open("knowledge.pdf", "rb"),
    purpose="assistants"
)

# Create assistant with file search
assistant = client.beta.assistants.create(
    name="My Assistant",
    instructions="You are a helpful assistant",
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
)

# Query
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="How do I configure voice?"
)
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)
```

### Pros

- ✅ **Zero setup** - Upload files, done
- ✅ **Managed infrastructure** - No servers to maintain
- ✅ **Automatic updates** - Improvements without code changes

### Cons

- ❌ **Vendor lock-in** - Entirely dependent on OpenAI
- ❌ **No control over chunking** - Black box processing
- ❌ **No control over search** - Can't tune parameters
- ❌ **Expensive** - Vector storage + LLM inference costs add up
- ❌ **Latency** - Multiple API calls (create thread, message, run, retrieve)
- ❌ **Data privacy** - Your documents live on OpenAI servers
- ❌ **Limited to OpenAI models** - Can't use other LLMs

### Cost Comparison

**Example:** 1,000 queries/day using GPT-4

OpenAI Assistants:
- Vector storage: $0.10/GB/day (~$3-30/month depending on doc size)
- GPT-4 inference: ~$0.03-0.06 per query (input + output tokens)
- 1,000 queries/day × $0.045 average = $1,350/month (LLM only)
- **Total: ~$1,400-1,500/month**

SignalWire Agents SDK:
- Self-hosted: ~$100/month (pgvector)
- Bring your own LLM (use any provider, same LLM costs apply separately)
- **Total: ~$100/month for search infrastructure**

**Verdict:** 15x more expensive for search infrastructure alone. Both approaches have LLM inference costs.

## LangChain + Pinecone/Weaviate/Qdrant

### How It Works

Use LangChain framework with external vector database:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

# Initialize Pinecone
pinecone.init(api_key="...", environment="...")
index = pinecone.Index("knowledge")

# Load and chunk documents
loader = DirectoryLoader('./docs')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(documents)

# Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="knowledge")

# Query
query = "How do I configure voice?"
results = vectorstore.similarity_search(query, k=5)
```

### Pros

- ✅ **Flexible** - Many integrations, customizable
- ✅ **Popular** - Large community, lots of examples
- ✅ **Choice of vector DBs** - Pinecone, Weaviate, Qdrant, etc.

### Cons

- ❌ **Complex setup** - Many moving parts
- ❌ **External dependencies** - Requires managed vector DB subscription
- ❌ **Additional costs** - Vector DB fees ($70+/month)
- ❌ **More code** - Manual integration with agent framework
- ❌ **Separate infrastructure** - Vector DB separate from agent
- ❌ **Latency** - Network calls to external service

### Code Complexity Comparison

**LangChain approach:**
```python
# Multiple services to configure
pinecone.init(...)
embeddings = OpenAIEmbeddings(openai_api_key=...)
vectorstore = Pinecone.from_documents(...)

# Manual retrieval
results = vectorstore.similarity_search(query)

# Manual formatting
context = "\n".join([r.page_content for r in results])

# Manual integration with agent
agent_prompt = f"Context: {context}\n\nQuestion: {query}"
```

**SignalWire approach:**
```python
# One skill addition
agent.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_path": "./knowledge.swsearch"
})

# Agent automatically uses it
# No manual retrieval, formatting, or integration
```

### Cost Comparison

**Pinecone pricing:**

- Starter: $70/month (up to 100K vectors)
- Standard: $0.096/GB/month + compute
- Enterprise: Custom pricing

**Plus:**

- OpenAI embeddings: $0.0001 per 1K tokens
- LLM inference costs
- Infrastructure for agent

**SignalWire SDK:**

- Self-hosted pgvector: ~$100/month (unlimited vectors)
- Or .swsearch files: $0 (included)

**Verdict:** Similar costs for small scale, SDK cheaper at scale.

## DIY: Embedding API + Custom Search

### How It Works

Build your own with OpenAI/Cohere embeddings:

```python
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Generate embeddings for documents
documents = ["Document 1 text...", "Document 2 text..."]
doc_embeddings = []

for doc in documents:
    response = openai.Embedding.create(
        input=doc,
        model="text-embedding-3-small"
    )
    doc_embeddings.append(response['data'][0]['embedding'])

# Store embeddings (manually)
# ... your storage solution ...

# Query
query = "How do I configure voice?"
query_response = openai.Embedding.create(
    input=query,
    model="text-embedding-3-small"
)
query_embedding = query_response['data'][0]['embedding']

# Search (manually)
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
top_k = np.argsort(similarities)[-5:][::-1]
results = [documents[i] for i in top_k]
```

### Pros

- ✅ **Full control** - Every aspect customizable
- ✅ **No framework** - Minimal dependencies
- ✅ **Learning experience** - Understand how it works

### Cons

- ❌ **Lots of code** - Implement everything yourself
- ❌ **Maintenance burden** - Updates, bug fixes on you
- ❌ **API costs** - Pay per embedding generation
- ❌ **Complexity** - Document loading, chunking, storage, retrieval
- ❌ **No optimizations** - Miss hybrid search, metadata boosting
- ❌ **Reinventing the wheel** - Solving solved problems

### Development Time Comparison

**DIY approach:**

- Document loading: 2-4 hours
- Chunking strategies: 4-8 hours
- Embedding generation: 2-4 hours
- Storage (database): 4-8 hours
- Search implementation: 4-8 hours
- Hybrid search: 8-16 hours
- Metadata filtering: 4-8 hours
- Testing and optimization: 8-16 hours
**Total: 36-72 hours**

**SignalWire SDK:**

- Add skill: 5 minutes
- Build index: 5 minutes
- Configure: 10 minutes
**Total: 20 minutes**

## Why Integrated Search Matters

The SignalWire approach integrates search directly into the agent framework. Here's why that matters:

### 1. No Additional Infrastructure

**Other approaches:**
```
Your Agent → External Vector DB → Results
   ↓              ↓
  LLM      (Pinecone/Weaviate/etc)
```

Requires:
- Vector DB subscription
- Network connectivity
- Connection management
- Separate monitoring

**SignalWire approach:**
```
Your Agent (includes search)
   ↓
Results
```

Everything in one place:
- No external dependencies (with .swsearch)
- Or shared pgvector (your infrastructure)
- Single monitoring surface

### 2. Zero Latency Overhead

**External vector DB:**

- Network request: 20-50ms
- Query processing: 10-30ms
- Response: 20-50ms
**Total added latency: 50-130ms**

**Integrated search:**

- Local SQLite: <5ms
- Local pgvector: 10-30ms
**Total: 5-30ms**

For voice agents where latency matters, this is significant.

### 3. Automatic Tool Integration

**LangChain approach:**
```python
# Manual retrieval
docs = vectorstore.similarity_search(query)

# Manual context building
context = format_docs(docs)

# Manual prompt engineering
prompt = f"Given: {context}\n\nAnswer: {query}"

# Manual LLM call
response = llm.generate(prompt)
```

**SignalWire approach:**
```python
# Just add skill - agent automatically:
# - Calls search when needed
# - Formats context
# - Includes in prompt
# - Generates response
agent.add_skill("native_vector_search", {...})
```

The agent framework handles the orchestration.

### 4. Optimized for Voice

Other solutions are text-focused. SignalWire SDK considers voice:

```python
# Automatically adapts to voice vs chat
def _format_search_results(self, response, agent, query, results, **kwargs):
    is_voice = getattr(agent, 'is_voice', False)

    if is_voice:
        # Don't read URLs, summarize code
        return format_for_voice(results)
    else:
        # Include links, show code blocks
        return format_for_chat(results)
```

Built-in voice optimization.

### 5. Portable Deployments

**.swsearch files are self-contained:**
```bash
# Build once
sw-search ./docs --output knowledge.swsearch

# Deploy anywhere
scp knowledge.swsearch lambda:/var/task/
scp knowledge.swsearch docker-container:/app/
scp knowledge.swsearch edge-device:/opt/agent/
```

No external services required.

### 6. Cost Efficiency

**Monthly costs for 10,000 queries:**

| Solution | Search Infrastructure Cost | Notes |
|----------|---------------------------|-------|
| OpenAI Assistants | ~$1,400/mo | Vector storage + GPT-4 inference |
| LangChain + Pinecone | $70-200/mo | Vector DB subscription + API fees |
| DIY + OpenAI Embeddings | $50-150/mo | Embedding API + your infrastructure |
| SignalWire SDK (.swsearch) | $0/mo | Included, no external deps |
| SignalWire SDK (pgvector) | ~$100/mo | Self-hosted PostgreSQL |

**Note:** All approaches have LLM inference costs. Comparison shows search infrastructure costs only.

## Feature Comparison Matrix

| Feature | OpenAI Assistants | LangChain + VectorDB | DIY | SignalWire SDK |
|---------|-------------------|----------------------|-----|----------------|
| **Setup Time** | 30 min | 2-4 hours | 8-16 hours | 20 min |
| **Chunking Control** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes (9 strategies) |
| **Hybrid Search** | ❓ Unknown | ⚠️ Manual | ⚠️ Manual | ✅ Built-in |
| **Metadata Filtering** | ❌ Limited | ✅ Yes | ⚠️ Manual | ✅ Yes |
| **Cost (10K queries)** | ~$1,400/mo | $100-200/mo | $50-150/mo | $0-100/mo |
| **Vendor Lock-in** | ❌ High | ⚠️ Moderate | ✅ None | ✅ None |
| **Offline Operation** | ❌ No | ❌ No | ⚠️ Possible | ✅ Yes (.swsearch) |
| **Voice Optimized** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Agent Integration** | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ✅ Automatic |
| **Deployment Size** | N/A | Large | Medium | Small (query-only) |
| **Latency** | 100-200ms | 50-130ms | 50-100ms | 5-30ms |
| **LLM Choice** | OpenAI only | Any | Any | Any |
| **Data Privacy** | ❌ OpenAI servers | ⚠️ 3rd party | ✅ Your infra | ✅ Your infra |
| **Scalability** | ✅ Auto | ✅ Managed | ⚠️ DIY | ✅ pgvector |

## When to Use What

### Use OpenAI Assistants If:
- You're prototyping quickly
- Budget isn't a concern
- You're already all-in on OpenAI
- Data privacy isn't critical

### Use LangChain + VectorDB If:
- You need maximum flexibility
- You're comfortable with complexity
- You want to use multiple LLMs
- You need integrations LangChain provides

### Use DIY If:
- You need absolute control
- You're building something unique
- You want to learn the internals
- Cost optimization is critical

### Use SignalWire SDK If:
- You're building voice agents
- You want simple, fast setup
- You want portability
- Cost efficiency matters
- You want integrated solution
- You need production-ready RAG

## Migration Paths

### From OpenAI Assistants

1. Export your documents
2. Build .swsearch index: `sw-search ./docs --output kb.swsearch`
3. Add skill: `agent.add_skill("native_vector_search", {...})`
4. Significantly reduce search infrastructure costs

### From LangChain + Pinecone

1. Keep your documents (already have them)
2. Build index with same chunking: `sw-search ./docs --chunking-strategy ...`
3. Migrate Pinecone data if needed: `sw-search migrate ...`
4. Replace LangChain code with skill
5. Cancel Pinecone subscription

### From DIY Solution

1. You already have chunking logic
2. Export to JSON format
3. Build index: `sw-search ./chunks.json --chunking-strategy json`
4. Replace search code with skill
5. Reduce maintenance burden

## Key Takeaways

1. **Self-hosted is cost-effective** - $0-100/mo vs $1,400+/mo for managed solutions
2. **External vector DBs add latency** - 50-130ms vs 5-30ms
3. **Integration matters** - automatic vs manual orchestration
4. **SDK optimized for voice** - adaptive formatting built-in
5. **Portability unique** - .swsearch files work anywhere
6. **Setup time dramatically different** - 20 min vs 8+ hours
7. **Choose based on priorities** - cost, control, simplicity
8. **LLM costs are separate** - All approaches have similar inference costs

The SignalWire Agents SDK search system delivers production-ready RAG with minimal setup, maximum portability, and cost efficiency - purpose-built for conversational AI agents.

Next, we'll explore performance and optimization techniques to get the most out of your search deployment.
# 17. Performance and Optimization

Performance matters for production deployments. Whether you're building a voice agent where latency is critical or scaling to thousands of queries per day, understanding performance characteristics helps you optimize your search system.

## Index Build Performance

Building indexes is a one-time (or periodic) operation. Understanding the costs helps with CI/CD planning.

### Build Time Factors

**1. Content volume:**

- 100 documents (~1MB): 30-60 seconds
- 1,000 documents (~10MB): 5-10 minutes
- 10,000 documents (~100MB): 30-60 minutes

**2. Chunking strategy:**

- Simple (sentence, paragraph): Fastest
- Structural (markdown): Moderate (needs parsing)
- AI-based (semantic, topic): Slowest (requires inference)

**3. Embedding model:**

- Mini (384 dims): ~1,000 chunks/second
- Base (768 dims): ~500 chunks/second
- Large (1024 dims): ~200 chunks/second

**4. Hardware:**

- CPU: Significant impact (embedding generation)
- Memory: Need ~2GB + index size
- Disk: I/O speed matters for pgvector

### Build Time Benchmarks

**Small knowledge base (100 docs, 2,000 chunks):**
```bash
# Mini model, sentence chunking
sw-search ./docs --model mini --chunking-strategy sentence --output docs.swsearch
# Time: 45 seconds
# Size: 8MB
```

**Medium knowledge base (1,000 docs, 20,000 chunks):**
```bash
# Mini model, markdown chunking
sw-search ./docs --model mini --chunking-strategy markdown --output docs.swsearch
# Time: 8 minutes
# Size: 80MB
```

**Large knowledge base (10,000 docs, 200,000 chunks):**
```bash
# Base model, markdown chunking, pgvector
sw-search ./docs --model base --chunking-strategy markdown \
  --backend pgvector --connection-string "$PG_CONN" \
  --collection-name docs
# Time: 45 minutes
# Database size: 500MB
```

### Optimizing Build Performance

**1. Use mini model when quality allows:**
```bash
# 2-3x faster than base
sw-search ./docs --model mini --output docs.swsearch
```

**2. Choose efficient chunking:**
```bash
# Sentence is fastest
sw-search ./docs --chunking-strategy sentence --output docs.swsearch

# Markdown slightly slower but better for tech docs
sw-search ./docs --chunking-strategy markdown --output docs.swsearch
```

**3. Parallel processing (coming soon):**
```bash
# Process multiple directories in parallel
sw-search ./docs1 --output docs1.swsearch &
sw-search ./docs2 --output docs2.swsearch &
wait
```

**4. Incremental updates:**
```bash
# Build small index, merge later (future feature)
sw-search ./new-docs --output new.swsearch
sw-search merge docs.swsearch new.swsearch --output merged.swsearch
```

### CI/CD Integration

**GitHub Actions example:**
```yaml
name: Build Search Index

on:
  push:
    paths:
      - 'docs/**'

jobs:
  build-index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: pip install signalwire-agents[search-full]

      - name: Build index
        run: |
          sw-search ./docs \
            --model mini \
            --chunking-strategy markdown \
            --output knowledge.swsearch
        timeout-minutes: 15

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: search-index
          path: knowledge.swsearch
```

Build time is predictable, cache indexes in CI artifacts.

## Query Performance

Query performance impacts user experience directly. Fast queries = better agent responsiveness.

### Query Time Breakdown

**SQLite backend (.swsearch files):**
```
Total query time: 15-30ms
├── Embedding generation: 5-10ms (depends on model)
├── Vector search: 3-8ms (SQLite)
├── Hybrid scoring: 2-5ms
└── Result formatting: 1-2ms
```

**pgvector backend:**
```
Total query time: 20-50ms
├── Embedding generation: 5-10ms (depends on model)
├── Network latency: 1-5ms (if remote)
├── Vector search: 10-25ms (PostgreSQL)
├── Hybrid scoring: 2-5ms
└── Result formatting: 1-2ms
```

### Performance by Index Size

**Small index (< 5,000 chunks):**

- SQLite: 10-20ms
- pgvector: 15-30ms

**Medium index (5,000 - 50,000 chunks):**

- SQLite: 20-40ms
- pgvector: 25-50ms

**Large index (50,000+ chunks):**

- SQLite: 40-80ms
- pgvector: 30-60ms (scales better)

**Key insight:** pgvector scales better with size due to optimized indexing.

### Embedding Model Impact

**Query embedding generation time:**

- Mini model: 5-8ms
- Base model: 10-15ms
- Large model: 20-30ms

For most queries, embedding generation is 30-50% of total time. Mini model provides significant speedup.

### Optimizing Query Performance

**1. Use mini model:**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search documentation",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "docs",
    "model_name": "mini"  # Faster queries
})
```

**2. Reduce result count:**
```python
{
    "count": 3  # Faster than 5 or 10
}
```

**3. Use appropriate threshold:**
```python
{
    "distance_threshold": 0.4  # Filters early, reduces processing
}
```

**4. For pgvector, add indexes:**
```sql
-- Optimize vector search
CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Optimize metadata filtering
CREATE INDEX ON knowledge_chunks USING gin (metadata jsonb_path_ops);
```

## Memory Usage

Understanding memory requirements helps with capacity planning.

### Index Build Memory

**SQLite (.swsearch):**

- Mini model: ~2GB peak (model loading + processing)
- Base model: ~3GB peak
- Large model: ~4GB peak

**pgvector:**

- Mini model: ~2GB peak (no index in memory)
- Base model: ~3GB peak
- Large model: ~4GB peak

Memory usage is mostly model size + processing buffer.

### Query Memory

**SQLite:**
```
Runtime memory: ~1.5GB
├── Embedding model: 1GB (mini), 2GB (base)
├── SQLite index: Loaded on-demand (~50-100MB)
└── Query processing: ~50MB
```

**pgvector:**
```
Runtime memory: ~1.5GB
├── Embedding model: 1GB (mini), 2GB (base)
└── Query processing: ~50MB
```

pgvector is more memory-efficient for queries (index stays in database).

### Optimizing Memory Usage

**1. Use query-only mode in production:**
```bash
# Don't load document processing libraries
pip install signalwire-agents[search-queryonly]
```

Saves ~400MB runtime memory.

**2. Use mini model:**
```python
# 1GB vs 2GB for base model
{"model_name": "mini"}
```

**3. Shared model across agents:**
```python
# In multi-agent deployment, one model instance shared
# Automatic in AgentServer
```

**4. Lazy loading:**
```python
# Models load on first query, not at startup
# No memory cost until search is used
```

## Caching Strategies

Caching improves performance for repeated queries.

### Query Embedding Cache

Cache generated embeddings for common queries:

```python
from functools import lru_cache

class CachedSearchAgent(AgentBase):
    def __init__(self):
        super().__init__(name="CachedAgent")
        self._embedding_cache = {}

    @lru_cache(maxsize=100)
    def _cached_search(self, query: str):
        """Cache search results for identical queries"""
        # Search skill call (automatically cached)
        pass
```

Built-in caching in search skill (future enhancement).

### Result Caching

For frequently asked questions:

```python
import hashlib
import time

class CachingAgent(AgentBase):
    def __init__(self):
        super().__init__(name="CachingAgent")
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour

    def _get_cache_key(self, query):
        return hashlib.md5(query.lower().encode()).hexdigest()

    def search_with_cache(self, query):
        key = self._get_cache_key(query)
        now = time.time()

        # Check cache
        if key in self._cache:
            result, timestamp = self._cache[key]
            if now - timestamp < self._cache_ttl:
                return result

        # Miss - do real search
        result = self.do_search(query)

        # Store in cache
        self._cache[key] = (result, now)

        return result
```

Useful for FAQ agents with repeated questions.

### Index Caching

**SQLite:** Entire index is a single file, OS file cache helps.

**pgvector:** PostgreSQL has its own cache (shared_buffers):

```sql
-- Increase PostgreSQL cache for better performance
-- In postgresql.conf:
shared_buffers = 4GB
effective_cache_size = 12GB
```

## Scaling Strategies

### Horizontal Scaling

**Multiple agent instances, shared pgvector:**

```
Load Balancer
├── Agent Instance 1 ──┐
├── Agent Instance 2 ──┼──> PostgreSQL (pgvector)
├── Agent Instance 3 ──┤
└── Agent Instance 4 ──┘
```

All instances query same index. No duplication.

**Deployment:**
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: search-agent
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: agent
        image: my-agent:latest
        env:
        - name: PGVECTOR_CONNECTION
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: connection-string
```

Scales to hundreds of concurrent queries.

### Read Replicas

**For high query volume:**

```
Write: Main PostgreSQL ──> Index updates
                            │
Read:  ┌───────────────────┴────────────┐
       Agent 1 → Replica 1               │
       Agent 2 → Replica 2               │
       Agent 3 → Replica 1               │
       Agent 4 → Replica 2               │
```

Distribute query load across replicas.

### Sharding by Collection

**Large deployments with many collections:**

```
Agent determines collection:
├── User docs → DB1 (user_collection)
├── API docs → DB2 (api_collection)
└── Pricing → DB3 (pricing_collection)
```

Each database handles subset of collections.

### Connection Pooling

**For pgvector efficiency:**

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Connection pool for efficiency
engine = create_engine(
    os.getenv("PGVECTOR_CONNECTION"),
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# Use in agent configuration
connection_string = os.getenv("PGVECTOR_CONNECTION") + "?pool=true"
```

Reuses connections across queries.

## Performance Monitoring

### Metrics to Track

**1. Query latency:**
```python
import time

def monitored_search(query):
    start = time.time()
    results = search(query)
    latency = time.time() - start
    logger.info(f"Search latency: {latency*1000:.1f}ms")
    return results
```

**2. Cache hit rate:**
```python
cache_hits = 0
cache_misses = 0

def track_cache():
    hit_rate = cache_hits / (cache_hits + cache_misses)
    logger.info(f"Cache hit rate: {hit_rate*100:.1f}%")
```

**3. Result quality:**
```python
def track_quality(query, results):
    if not results:
        logger.warning(f"No results for: {query}")

    avg_similarity = sum(r['similarity'] for r in results) / len(results)
    logger.info(f"Average similarity: {avg_similarity:.3f}")
```

**4. Error rate:**
```python
errors = 0
queries = 0

def track_errors():
    error_rate = errors / queries
    logger.info(f"Error rate: {error_rate*100:.2f}%")
```

### Production Dashboard

**Example metrics:**
```python
from prometheus_client import Counter, Histogram

# Prometheus metrics
search_queries = Counter('search_queries_total', 'Total search queries')
search_latency = Histogram('search_latency_seconds', 'Search query latency')
search_errors = Counter('search_errors_total', 'Search errors')
no_results = Counter('search_no_results_total', 'Queries with no results')

def monitored_search(query):
    search_queries.inc()

    with search_latency.time():
        try:
            results = search(query)
            if not results:
                no_results.inc()
            return results
        except Exception as e:
            search_errors.inc()
            raise
```

## Optimization Checklist

### For Query Performance:
- [ ] Use mini model when quality is sufficient
- [ ] Reduce result count (3-5 instead of 10+)
- [ ] Set appropriate distance_threshold (0.4-0.5)
- [ ] Use pgvector for large indexes
- [ ] Add indexes to pgvector collections
- [ ] Implement connection pooling

### For Memory Efficiency:
- [ ] Use search-queryonly in production
- [ ] Use mini model (1GB vs 2GB)
- [ ] Use pgvector (index not in memory)
- [ ] Lazy load search models

### For Scalability:
- [ ] Use pgvector for shared access
- [ ] Deploy multiple agent instances
- [ ] Implement connection pooling
- [ ] Consider read replicas for high volume
- [ ] Monitor query performance

### For Build Performance:
- [ ] Use mini model for faster builds
- [ ] Choose efficient chunking strategy
- [ ] Cache indexes in CI/CD
- [ ] Build incrementally when possible

## Real-World Performance Example

**Sigmond in production:**

**Setup:**

- 3 collections (7,500 total chunks)
- 4 agent instances (Kubernetes)
- pgvector (single database)
- Mini model
- 1,000+ queries/day

**Performance:**

- Average query time: 25ms
- P95 query time: 45ms
- Memory per agent: 1.5GB
- Cache hit rate: 35%
- Error rate: <0.1%

**Cost:**

- Database: $100/month
- 4 agent instances: $200/month
- Total: $300/month

Handles production load efficiently.

## Key Takeaways

1. **Query performance: 15-50ms** - fast enough for voice agents
2. **Mini model is fastest** - 2-3x faster than base, minimal quality loss
3. **pgvector scales better** - for large indexes and multiple agents
4. **Memory: ~1.5GB per agent** - mostly embedding model
5. **search-queryonly saves memory** - 400MB reduction in production
6. **Caching helps** - 30-40% hit rate typical for FAQ agents
7. **Horizontal scaling works** - multiple agents, shared pgvector
8. **Monitor key metrics** - latency, error rate, no-result rate
9. **Build times predictable** - cache indexes in CI/CD
10. **Cost-effective at scale** - $100-300/month handles thousands of queries

The search system is optimized for production use - fast queries, efficient memory usage, and horizontal scalability for high-volume deployments.

Next, we'll explore troubleshooting common issues and how to diagnose and fix them.
# 18. Troubleshooting Common Issues

Even with a well-designed system, issues can arise. This section covers common problems, their symptoms, diagnosis steps, and solutions.

## Issue 1: No Results Returned

### Symptoms

```python
Query: "how to authenticate"
Results: []
```

The search returns empty results for queries that should match your content.

### Diagnosis

**Step 1: Verify the index exists and has content**
```bash
# Check if index file exists
ls -lh knowledge.swsearch

# Validate index
sw-search validate ./knowledge.swsearch
```

**Step 2: Test query directly**
```bash
# Try the same query via CLI
sw-search search ./knowledge.swsearch "how to authenticate" --verbose
```

Look at similarity scores in verbose output.

**Step 3: Check distance threshold**
```python
# Is threshold too strict?
{
    "distance_threshold": 0.7  # Very strict
}
```

### Common Causes

**1. Distance threshold too strict**
```python
# Problem: threshold too high
{
    "distance_threshold": 0.7  # Only near-perfect matches
}

# Solution: lower threshold
{
    "distance_threshold": 0.4  # More permissive
}
```

**2. Content doesn't exist**
```bash
# Check what's actually in the index
sw-search search ./knowledge.swsearch "authentication" --verbose --count 10

# If truly no content about authentication, add it
sw-search ./auth-docs --output updated.swsearch
```

**3. Query phrasing doesn't match content**
```python
# Query: "how to authenticate"
# Content: "bearer token usage"

# These might not match semantically
# Solution: Try synonyms
sw-search search ./knowledge.swsearch "bearer token" --verbose
```

**4. Model mismatch**
```python
# Built with base model
sw-search ./docs --model base --output docs.swsearch

# But querying with mini model
{
    "model_name": "mini"  # WRONG MODEL
}

# Solution: match the model
{
    "model_name": "base"
}
```

### Solutions

**Adjust threshold dynamically:**
```python
def search_with_fallback(self, query):
    """Try multiple thresholds"""
    for threshold in [0.5, 0.4, 0.3]:
        results = self.search(query, threshold=threshold)
        if results:
            return results
    return []  # Truly no results
```

**Add more content:**
```bash
# If content gap identified, add more docs
sw-search ./additional-docs --output additional.swsearch

# Merge indexes (future feature) or rebuild
sw-search ./all-docs --output complete.swsearch
```

**Improve query understanding:**
```python
# Expand query with synonyms
def expand_query(query):
    synonyms = {
        "authenticate": ["login", "sign in", "bearer token", "credentials"],
        "configure": ["setup", "set up", "initialize", "configure"]
    }
    # Search with expanded terms
```

## Issue 2: Irrelevant Results

### Symptoms

```python
Query: "Python authentication examples"
Results:
1. "Ruby authentication guide" (similarity: 0.65)
2. "General API overview" (similarity: 0.58)
3. "Installation instructions" (similarity: 0.52)
```

Results are returned but don't match the query intent.

### Diagnosis

**Step 1: Check similarity scores**
```bash
sw-search search ./knowledge.swsearch "Python authentication examples" --verbose
```

Are scores above 0.5? Lower scores = weaker matches.

**Step 2: Examine result content**
```bash
# Look at what's actually being returned
sw-search search ./knowledge.swsearch "Python authentication examples" --verbose --count 10
```

**Step 3: Check chunking quality**
```bash
# Export to see chunks
sw-search export ./knowledge.swsearch ./exported.json

# Look at chunk boundaries - are they mixing topics?
```

### Common Causes

**1. Threshold too permissive**
```python
{
    "distance_threshold": 0.2  # Too low, accepts weak matches
}

# Solution: raise threshold
{
    "distance_threshold": 0.4
}
```

**2. Poor chunking strategy**
```python
# Problem: using sentence strategy on code docs
sw-search ./docs --chunking-strategy sentence  # Splits code blocks

# Solution: use markdown strategy
sw-search ./docs --chunking-strategy markdown  # Preserves code
```

**3. Missing metadata/tags**
```bash
# Content has no tags to boost relevance
# Solution: add tags during indexing
sw-search ./docs --tags python,authentication,examples --output docs.swsearch
```

**4. Content genuinely lacks specificity**
```
# Query: "Python authentication"
# Content: Only has generic auth discussion, no Python-specific examples

# Solution: Add Python-specific content
```

### Solutions

**Increase threshold:**
```python
{
    "distance_threshold": 0.5  # Stricter matching
}
```

**Use tag filtering:**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_python_examples",
    "description": "Search for Python code examples",
    "index_path": "./docs.swsearch",
    "tags": ["python", "code"],  # Only Python code chunks
    "distance_threshold": 0.4
})
```

**Rebuild with better chunking:**
```bash
# Use markdown strategy for technical docs
sw-search ./docs \
  --chunking-strategy markdown \
  --tags documentation,api \
  --output docs_improved.swsearch
```

**Add custom metadata:**
```json
{
  "chunks": [
    {
      "content": "Here's how to authenticate in Python...",
      "metadata": {
        "language": "python",
        "category": "authentication",
        "tags": ["python", "auth", "example", "code"]
      }
    }
  ]
}
```

## Issue 3: Best Result Not First

### Symptoms

```python
Query: "voice configuration"
Results:
1. "General configuration overview" (similarity: 0.72)
2. "Database configuration" (similarity: 0.68)
3. "Perfect voice config guide" (similarity: 0.85)  ← Should be #1
```

The best result is buried in the list.

### Diagnosis

**Step 1: Check hybrid scoring**
```bash
# Look at verbose output
sw-search search ./knowledge.swsearch "voice configuration" --verbose
```

Check if result #3 has lower hybrid score despite higher vector similarity.

**Step 2: Examine metadata**
```bash
# Export and look at metadata
sw-search export ./knowledge.swsearch ./exported.json

# Does the best result have relevant tags?
```

### Common Causes

**1. Missing metadata on best result**
```json
// Best result has no helpful metadata
{
  "content": "Voice configuration guide...",
  "metadata": {}  // No tags to boost it
}

// Other results have metadata
{
  "content": "General configuration...",
  "metadata": {
    "tags": ["configuration", "setup"]  // Gets keyword boost
  }
}
```

**2. Keyword matches on wrong results**
```python
# Query: "voice configuration"
# Result #1: "Configuration for voice, database, and API..."
# → Contains both keywords, gets boosted even though less relevant
```

**3. Code tag on wrong result**
```python
# Query asks for documentation
# Result with code gets +20% boost even though doc is better
```

### Solutions

**Add metadata to best result:**
```bash
# Export, edit, rebuild
sw-search export ./knowledge.swsearch ./chunks.json

# Edit chunks.json to add metadata
{
  "content": "Voice configuration guide...",
  "metadata": {
    "tags": ["voice", "configuration", "guide"],
    "category": "voice"
  }
}

# Rebuild
sw-search ./chunks.json --chunking-strategy json --output fixed.swsearch
```

**Adjust result count:**
```python
{
    "count": 5  # Return top 5, LLM will prioritize best
}
```

**Use custom formatter to reorder:**
```python
def _format_with_reranking(self, response, agent, query, results, **kwargs):
    """Rerank based on custom logic"""
    # Put code examples first for "example" queries
    if "example" in query.lower():
        results.sort(key=lambda r: r.get('metadata', {}).get('has_code', False), reverse=True)

    # Rebuild response with reordered results
    return self._build_response(results)
```

## Issue 4: Response Truncated

### Symptoms

```
Results look cut off mid-sentence...
```

Search results are incomplete or cut off.

### Diagnosis

**Step 1: Check content length budget**
```python
{
    "max_content_length": 16384  # Is this too small?
}
```

**Step 2: Look at chunk sizes**
```bash
# Export and examine chunk lengths
sw-search export ./knowledge.swsearch ./chunks.json

# Check if individual chunks are very long
```

**Step 3: Check result count**
```python
{
    "count": 10  # Too many results for the budget
}
```

### Common Causes

**1. max_content_length too low**
```python
{
    "max_content_length": 8192  # Too small for 5 results
}
```

**2. Individual chunks too long**
```bash
# Using page strategy on long documents
sw-search ./docs --chunking-strategy page  # Creates huge chunks
```

**3. Too many results requested**
```python
{
    "count": 10,  # 10 results
    "max_content_length": 16384  # Only ~1.6KB per result
}
```

### Solutions

**Increase budget:**
```python
{
    "max_content_length": 32768  # 32KB (default)
}
```

**Reduce result count:**
```python
{
    "count": 3,  # Fewer results = more space each
    "max_content_length": 16384
}
```

**Rechunk with smaller chunks:**
```bash
# Use sentence or paragraph instead of page
sw-search ./docs \
  --chunking-strategy sentence \
  --max-chunk-size 1000 \
  --output docs_small_chunks.swsearch
```

## Issue 5: Search Function Not Called

### Symptoms

Agent doesn't use the search function even when it should.

### Diagnosis

**Step 1: Check function is registered**
```bash
# List available tools
swaig-test agent.py --list-tools

# Should see search_docs or similar
```

**Step 2: Check function description**
```python
{
    "tool_name": "search_docs",
    "description": "Search documentation"  # Too vague?
}
```

**Step 3: Check prompt instructions**
```python
# Does agent know when to use search?
```

### Common Causes

**1. Vague function description**
```python
# Problem: too generic
{
    "description": "Search docs"
}

# Solution: be specific
{
    "description": "Search technical documentation for information about API endpoints, authentication, configuration, and troubleshooting. Use this when the user asks technical questions about the platform."
}
```

**2. No prompt instructions**
```python
# Problem: agent not told to use search
# No instructions

# Solution: explicit guidance
self.prompt_add_section(
    "Using Search",
    bullets=[
        "ALWAYS search the knowledge base before answering technical questions",
        "Use search_docs for API, configuration, and how-to questions",
        "Base answers on search results, not general knowledge"
    ]
)
```

**3. LLM thinks it knows the answer**
```python
# For common questions, LLM might not search
# Solution: stronger prompt
"ALWAYS search before answering, even if you think you know the answer"
```

### Solutions

**Improve function description:**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_product_docs",
    "description": (
        "Search comprehensive product documentation. "
        "Use this for ANY question about: features, configuration, "
        "API usage, troubleshooting, code examples, or technical details. "
        "This is your primary source of truth."
    ),
    "index_path": "./docs.swsearch"
})
```

**Add prompt instructions:**
```python
self.prompt_add_section(
    "Search First Policy",
    bullets=[
        "Before answering ANY technical question, search the knowledge base",
        "Use search_product_docs for product-related questions",
        "Never guess or use general knowledge - always search first",
        "If search returns no results, tell user you don't have that information"
    ]
)
```

**Test function directly:**
```bash
# Verify it works
swaig-test agent.py --exec search_docs --query "test"
```

## Issue 6: Model Loading Errors

### Symptoms

```
Error loading model: sentence-transformers/all-MiniLM-L6-v2
```

### Diagnosis

**Step 1: Check installation**
```bash
pip list | grep sentence-transformers
```

**Step 2: Check model cache**
```bash
ls ~/.cache/huggingface/transformers/
```

**Step 3: Check network access**
```bash
# Can you reach Hugging Face?
curl -I https://huggingface.co
```

### Common Causes

**1. Missing dependencies**
```bash
# Problem: search dependencies not installed
pip install signalwire-agents  # Base, no search

# Solution: install search
pip install signalwire-agents[search]
```

**2. Network issues**
```bash
# Problem: can't download model on first run
# Solution: pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**3. Disk space**
```bash
# Models are ~400MB
df -h ~/.cache

# Clean if needed
rm -rf ~/.cache/huggingface/transformers/
```

### Solutions

**Pre-download models:**
```dockerfile
# In Docker build
RUN pip install signalwire-agents[search] && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Use offline mode:**
```bash
# Download models beforehand
# Set environment variable
export TRANSFORMERS_OFFLINE=1
```

**Use query-only mode:**
```bash
# If only querying existing indexes
pip install signalwire-agents[search-queryonly]
```

## Issue 7: pgvector Connection Errors

### Symptoms

```
Error: could not connect to PostgreSQL server
```

### Diagnosis

**Step 1: Test connection**
```bash
psql "$PGVECTOR_CONNECTION"
```

**Step 2: Check connection string format**
```python
# Should be: postgresql://user:pass@host:port/db
connection_string = os.getenv("PGVECTOR_CONNECTION")
print(connection_string)  # Check format
```

**Step 3: Check pgvector extension**
```sql
-- Connect to database
\dx

-- Should see vector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### Common Causes

**1. Wrong connection string**
```python
# Problem: missing components
"postgresql://localhost/db"  # Missing user, pass, port

# Solution: complete string
"postgresql://user:pass@localhost:5432/db"
```

**2. pgvector extension not installed**
```sql
-- Check
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Install if missing
CREATE EXTENSION vector;
```

**3. Network/firewall issues**
```bash
# Can you reach the host?
telnet postgres-host 5432
```

**4. Authentication failure**
```bash
# Check credentials
psql -h host -p 5432 -U user -d db
```

### Solutions

**Verify connection string:**
```python
import os
conn = os.getenv("PGVECTOR_CONNECTION")
assert conn.startswith("postgresql://"), "Invalid connection string"
assert "@" in conn, "Missing credentials"
assert ":" in conn.split("@")[1], "Missing port"
```

**Install pgvector:**
```sql
-- As superuser
CREATE EXTENSION vector;

-- Grant access
GRANT ALL ON SCHEMA public TO your_user;
```

**Test connection:**
```python
from sqlalchemy import create_engine

engine = create_engine(os.getenv("PGVECTOR_CONNECTION"))
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print("Connected successfully")
```

## Issue 8: Slow Queries

### Symptoms

Search queries taking 500ms+ instead of 20-50ms.

### Diagnosis

**Step 1: Profile query**
```bash
# Test query time
time sw-search search ./knowledge.swsearch "test query"
```

**Step 2: Check index size**
```bash
# Large index?
ls -lh knowledge.swsearch

# Or for pgvector
SELECT count(*) FROM knowledge_chunks;
```

**Step 3: Check model**
```python
{
    "model_name": "large"  # Slow model?
}
```

### Common Causes

**1. Using large model**
```python
{
    "model_name": "large"  # 3x slower than mini
}
```

**2. No pgvector indexes**
```sql
-- Check for indexes
\d knowledge_chunks

-- If no vector index, queries are slow
```

**3. Too many results**
```python
{
    "count": 20  # Requesting too many
}
```

**4. Network latency**
```python
# Remote pgvector with high latency
"postgresql://user:pass@remote-host:5432/db"
```

### Solutions

**Switch to mini model:**
```python
{
    "model_name": "mini"  # 2-3x faster
}
```

**Add pgvector indexes:**
```sql
CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

**Reduce result count:**
```python
{
    "count": 3  # Faster than 10+
}
```

**Use local SQLite:**
```python
# For single-agent deployments
{
    "index_path": "./local.swsearch"  # Faster than remote pgvector
}
```

## Debugging Checklist

When troubleshooting, work through this checklist:

- [ ] Verify index exists and validates
- [ ] Test query with CLI tool (sw-search)
- [ ] Check distance threshold (0.4-0.5 typical)
- [ ] Verify model matches (build vs query)
- [ ] Examine chunk quality (export and review)
- [ ] Check function description is clear
- [ ] Add prompt instructions for search usage
- [ ] Monitor search latency
- [ ] Check for missing dependencies
- [ ] Test pgvector connection if used
- [ ] Review verbose output for similarity scores
- [ ] Verify metadata and tags are present

## Getting Help

**1. Enable verbose logging:**
```python
import os
os.environ['SIGNALWIRE_LOG_LEVEL'] = 'DEBUG'
```

**2. Test with CLI:**
```bash
sw-search search ./knowledge.swsearch "query" --verbose
```

**3. Export and inspect:**
```bash
sw-search export ./knowledge.swsearch ./inspect.json
```

**4. Check GitHub issues:**
```
https://github.com/signalwire/signalwire-agents-sdk/issues
```

## Key Takeaways

1. **No results = threshold too strict** - lower from 0.5 to 0.4 or 0.3
2. **Irrelevant results = threshold too permissive** - raise from 0.3 to 0.4
3. **Poor ranking = missing metadata** - add tags to boost relevance
4. **Truncation = budget too small** - increase max_content_length or reduce count
5. **Function not called = vague description** - be specific about when to use
6. **Model errors = missing dependencies** - install search extras
7. **Connection errors = check connection string** - verify format and credentials
8. **Slow queries = wrong model or no indexes** - use mini model, add pgvector indexes
9. **Always test with CLI first** - isolate agent vs search issues
10. **Export and inspect chunks** - understand what's actually indexed

Most issues are configuration-related and easily fixed. The CLI tools and verbose logging are your best friends for diagnosis.

Next, we'll wrap up with a conclusion that ties everything together and provides guidance for building production-ready search-powered agents.
# 19. Conclusion: Building Smarter Agents

We've covered a lot of ground - from understanding vector embeddings to deploying production search systems. Let's tie it all together and give you a roadmap for building search-powered agents.

## What We've Learned

### The Problem and Solution

**The problem:** LLMs hallucinate. They generate plausible-sounding but incorrect information when they don't know the answer. For production agents, this is unacceptable.

**The solution:** RAG (Retrieval-Augmented Generation) - give your agent access to real information through search, so it answers based on facts, not guesses.

**The SignalWire approach:** Integrated search that's self-contained, production-ready, and optimized for conversational AI.

### Core Concepts

**1. Vector Search**

- Text becomes embeddings (numeric vectors)
- Similar meaning = nearby vectors
- Cosine similarity measures relevance
- Works across languages and phrasings

**2. Hybrid Search**

- Starts with vector search (semantic understanding)
- Boosts with keyword matching (confirmation)
- Metadata provides additional signals
- Best of both worlds

**3. Chunking Strategies**

- Break content into searchable pieces
- 9 strategies for different content types
- Markdown strategy best for technical docs
- JSON workflow for manual curation

**4. Deployment Options**

- SQLite (.swsearch) for single agents
- pgvector for multi-agent, production scale
- query-only mode for minimal footprint
- All three work seamlessly

### The Tools

**sw-search CLI:**

- Build indexes from documents
- Validate, search, export, migrate
- Multiple backends supported
- Production-ready tooling

**native_vector_search skill:**

- Drop-in agent integration
- Configurable behavior
- Custom formatters
- Multiple skills per agent

**Three install modes:**

- `[search-queryonly]` - query only (~100MB)
- `[search-full]` - build + query (~600MB)
- `[search-all]` - everything (~700MB)

## Building Your First Search Agent

Here's the quickest path to a working search agent:

### Step 1: Install (2 minutes)

```bash
# For building indexes
pip install signalwire-agents[search-full]
```

### Step 2: Build Index (5 minutes)

```bash
# From your documentation
sw-search ./docs \
  --model mini \
  --chunking-strategy markdown \
  --output knowledge.swsearch
```

### Step 3: Create Agent (5 minutes)

```python
from signalwire_agents import AgentBase

class DocsAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="DocsAgent",
            route="/docs"
        )

        # Add search
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search documentation for technical information",
            "index_path": "./knowledge.swsearch"
        })

        # Tell agent to use search
        self.prompt_add_section(
            "Instructions",
            bullets=[
                "Always search before answering technical questions",
                "Base answers on search results",
                "If no results, say you don't have that information"
            ]
        )

if __name__ == "__main__":
    agent = DocsAgent()
    agent.serve(port=3000)
```

### Step 4: Test (2 minutes)

```bash
# Test search function
swaig-test agent.py --exec search_docs --query "your question"

# Run agent
python agent.py
```

**Total time: 15 minutes** from zero to working search agent.

## Moving to Production

Once you have a working agent, here's how to make it production-ready:

### 1. Optimize for Performance

```python
# Use mini model (2-3x faster, minimal quality loss)
sw-search ./docs --model mini --output knowledge.swsearch

# In agent
{
    "model_name": "mini",
    "count": 3,  # Fewer results = faster
    "distance_threshold": 0.4  # Balanced
}
```

### 2. Switch to pgvector for Scale

```bash
# Build to PostgreSQL
export PGVECTOR_CONNECTION="postgresql://user:pass@localhost:5432/db"

sw-search ./docs \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs \
  --model mini
```

```python
# In agent
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search documentation",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "docs",
    "model_name": "mini"
})
```

### 3. Use Query-Only Mode

```dockerfile
# Production container
FROM python:3.11-slim

# Install without heavy dependencies
RUN pip install signalwire-agents[search-queryonly]

COPY agent.py /app/
COPY knowledge.swsearch /app/  # Or use pgvector

CMD ["python", "/app/agent.py"]
```

Saves ~400MB of dependencies.

### 4. Multiple Collections

```python
# Different knowledge domains
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search general documentation",
    "collection_name": "general_docs"
})

self.add_skill("native_vector_search", {
    "tool_name": "search_api",
    "description": "Search API reference",
    "collection_name": "api_docs",
    "tags": ["api", "reference"]
})

self.add_skill("native_vector_search", {
    "tool_name": "search_troubleshooting",
    "description": "Search troubleshooting guides",
    "collection_name": "troubleshooting",
    "tags": ["errors", "troubleshooting"]
})
```

LLM chooses the right search for each question.

### 5. Custom Formatting

```python
def _format_search_results(self, response, agent, query, results, **kwargs):
    """Adapt to voice vs chat"""
    is_voice = getattr(agent, 'is_voice', False)

    if is_voice:
        # Conversational, no URLs
        instructions = "Provide a natural answer. Don't read URLs or code verbatim."
    else:
        # Include links and code
        instructions = "Include relevant URLs and format code with markdown."

    return instructions + "\n\n" + response
```

### 6. Monitor Performance

```python
import time

def _monitored_format(self, response, agent, query, results, **kwargs):
    """Track search metrics"""
    search_time = kwargs.get('search_time', 0)
    logger.info(f"Query: {query}")
    logger.info(f"Results: {len(results)}")
    logger.info(f"Time: {search_time*1000:.1f}ms")

    if not results:
        logger.warning(f"No results for: {query}")

    return response
```

## Common Patterns and Best Practices

### Pattern 1: FAQ Agent

```python
# Build from QA-structured content
sw-search ./faq \
  --chunking-strategy qa \
  --model mini \
  --output faq.swsearch

# Agent with strict threshold (exact matches)
{
    "distance_threshold": 0.4,
    "count": 3,
    "max_content_length": 16384  # Short, focused answers
}
```

### Pattern 2: Technical Documentation Agent

```python
# Build from markdown docs with code
sw-search ./docs \
  --chunking-strategy markdown \
  --model mini \
  --tags documentation,code \
  --output docs.swsearch

# Agent with moderate threshold
{
    "distance_threshold": 0.4,
    "count": 5,
    "max_content_length": 32768  # Detailed technical answers
}
```

### Pattern 3: Multi-Domain Support Agent

```python
# Multiple collections in pgvector
self.add_skill("native_vector_search", {
    "tool_name": "search_product",
    "description": "Search product documentation",
    "collection_name": "product_docs"
})

self.add_skill("native_vector_search", {
    "tool_name": "search_pricing",
    "description": "Search pricing information",
    "collection_name": "pricing"
})

self.add_skill("native_vector_search", {
    "tool_name": "search_company",
    "description": "Search company information",
    "collection_name": "company_info"
})
```

### Pattern 4: Voice Agent

```python
# Optimized for voice
{
    "max_content_length": 16384,  # Shorter responses
    "count": 3,  # Fewer results
    "response_format_callback": self._voice_format
}

def _voice_format(self, response, agent, query, results, **kwargs):
    """Voice-optimized formatting"""
    return (
        "Provide a concise, conversational answer. "
        "Don't read URLs, code, or technical identifiers verbatim. "
        "Summarize clearly for spoken delivery.\n\n" + response
    )
```

## Decision Guide

### Which Chunking Strategy?

- **Technical docs with code** → markdown
- **FAQ/support content** → qa
- **General articles/blogs** → paragraph
- **Academic papers** → semantic
- **Mixed content** → try markdown first
- **Custom needs** → JSON workflow

### Which Model?

- **Most use cases** → mini (fast, good quality)
- **High-quality needs** → base (better, slower)
- **Special cases** → large (research, complex)

**Recommendation:** Start with mini. Only upgrade if quality issues.

### Which Backend?

- **Single agent** → SQLite (.swsearch)
- **Multiple agents** → pgvector
- **Serverless** → SQLite (.swsearch)
- **High scale** → pgvector with replicas

### Which Install Mode?

- **Development** → `[search-full]`
- **Production** → `[search-queryonly]`
- **Building indexes** → `[search-full]`

## Success Metrics

Track these to measure search effectiveness:

**1. Usage rate**

- What % of queries use search?
- Target: 80%+ for technical questions
- If low: improve prompt instructions

**2. Result quality**

- What % of searches return results?
- Target: 95%+
- If low: lower distance_threshold or add content

**3. Average similarity**

- What's the typical similarity score?
- Target: 0.5-0.8
- If low: content doesn't match queries

**4. No-result queries**

- Which queries return nothing?
- Indicates content gaps
- Add missing content

**5. Query latency**

- How long do searches take?
- Target: <50ms
- If high: use mini model, add pgvector indexes

**6. User satisfaction**

- Do users get good answers?
- Track follow-up questions
- Measure resolution rate

## Common Mistakes to Avoid

**1. Not telling the agent to search**
```python
# ❌ Bad: no instructions
self.add_skill("native_vector_search", {...})

# ✅ Good: explicit instructions
self.prompt_add_section("Instructions", bullets=[
    "Always search before answering technical questions"
])
```

**2. Using wrong chunking strategy**
```python
# ❌ Bad: sentence strategy on code docs (splits code)
sw-search ./docs --chunking-strategy sentence

# ✅ Good: markdown preserves code blocks
sw-search ./docs --chunking-strategy markdown
```

**3. Threshold too strict or permissive**
```python
# ❌ Too strict: no results
{"distance_threshold": 0.7}

# ❌ Too permissive: irrelevant results
{"distance_threshold": 0.2}

# ✅ Balanced
{"distance_threshold": 0.4}
```

**4. Mismatched models**
```python
# ❌ Build with base, query with mini
sw-search ./docs --model base
{"model_name": "mini"}  # WRONG

# ✅ Match models
{"model_name": "base"}
```

**5. Ignoring metadata**
```python
# ❌ No tags, weak boosting
sw-search ./docs --output docs.swsearch

# ✅ Add tags for hybrid search
sw-search ./docs --tags documentation,api,code --output docs.swsearch
```

**6. Not testing before deploying**
```bash
# ❌ Deploy without testing
docker build && docker push

# ✅ Test first
sw-search search ./knowledge.swsearch "test queries"
swaig-test agent.py --exec search_docs --query "test"
```

## The Big Picture

Search transforms AI agents from confident guessers to reliable information sources. Here's why it matters:

**Without search:**

- Agent hallucinates answers
- Information becomes stale
- Can't answer specific questions
- No source of truth

**With search:**

- Agent cites real information
- Easy to update knowledge (rebuild index)
- Answers specific, detailed questions
- Search results are the source

**The SignalWire advantage:**

- Self-contained (no external services)
- Production-ready (fast, reliable)
- Portable (works anywhere)
- Cost-effective (60x cheaper than managed)

## Your Next Steps

**1. Start simple:**

- Install `[search-full]`
- Build index from your docs
- Add search skill to agent
- Test and iterate

**2. Tune for quality:**

- Try different chunking strategies
- Adjust distance threshold
- Add metadata and tags
- Test with real queries

**3. Scale to production:**

- Switch to pgvector
- Use query-only mode
- Deploy multiple instances
- Monitor performance

**4. Iterate based on metrics:**

- Track no-result queries
- Identify content gaps
- Adjust configuration
- Add missing content

## Resources

**CLI Reference:**
```bash
sw-search --help                    # Main help
sw-search search --help             # Search command help
sw-search export --help             # Export command help
```

**Example Agents:**
```
examples/
├── simple_search_agent.py         # Basic search integration
├── multi_collection_agent.py      # Multiple knowledge bases
└── production_agent.py            # Production configuration
```

**Documentation:**

- README.md - Installation and quickstart
- docs/search_system.md - Technical details
- docs/skills/native_vector_search.md - Skill reference

**GitHub:**

- Issues: https://github.com/signalwire/signalwire-agents-sdk/issues
- Discussions: https://github.com/signalwire/signalwire-agents-sdk/discussions

## Final Thoughts

Building AI agents that users trust requires grounding them in facts. Search is the foundation for that trust.

The SignalWire Agents SDK makes it simple:
- 15 minutes to working search agent
- 9 chunking strategies for any content
- SQLite or pgvector deployment
- Production-ready performance

You now have everything you need to build search-powered agents that:
- Answer accurately based on your content
- Scale to production workloads
- Adapt to voice and chat modes
- Update easily as content changes

Start with the basics, iterate based on real usage, and scale when you need to.

Welcome to the world of grounded, trustworthy AI agents.

---

**Ready to build?**

```bash
pip install signalwire-agents[search-full]
sw-search ./your-docs --output knowledge.swsearch
# Create your agent
# Deploy
```

It's that simple.
# Appendix A: Complete Configuration Reference

This appendix provides a comprehensive reference for all configuration options in the search system.

## sw-search CLI Reference

### Main Command

```bash
sw-search [SOURCE_PATHS...] [OPTIONS]
```

**Arguments:**

- `SOURCE_PATHS` - One or more paths to directories or files to index

### Build Options

**--output PATH**

- Output path for .swsearch file
- Required when building indexes
- Example: `--output knowledge.swsearch`

**--chunking-strategy STRATEGY**

- Chunking method to use
- Options: `sentence`, `paragraph`, `page`, `sliding_window`, `semantic`, `topic`, `qa`, `markdown`, `json`
- Default: `sentence`
- Example: `--chunking-strategy markdown`

**--model MODEL**

- Embedding model to use
- Options: `mini`, `base`, `large`
- Default: `mini`
- Models:
  - `mini`: all-MiniLM-L6-v2 (384 dimensions, fast)
  - `base`: all-mpnet-base-v2 (768 dimensions, balanced)
  - `large`: all-roberta-large-v1 (1024 dimensions, highest quality)
- Example: `--model mini`

**--backend BACKEND**

- Storage backend
- Options: `sqlite`, `pgvector`
- Default: `sqlite`
- Example: `--backend pgvector`

**--connection-string STRING**

- PostgreSQL connection string (for pgvector backend)
- Format: `postgresql://user:pass@host:port/database`
- Required when using pgvector
- Example: `--connection-string "postgresql://user:pass@localhost:5432/knowledge"`

**--collection-name NAME**

- Collection name in pgvector database
- Default: `default`
- Multiple collections can exist in one database
- Example: `--collection-name product_docs`

**--tags TAG1,TAG2,...**

- Comma-separated tags to add to all chunks
- Used for filtering and hybrid search boosting
- Example: `--tags documentation,api,python`

**--file-types EXT1,EXT2,...**

- Comma-separated file extensions to include
- Default: `md,txt,pdf,docx,html`
- Example: `--file-types md,txt,rst`

**--exclude-patterns PATTERN1,PATTERN2,...**

- Glob patterns for files to exclude
- Example: `--exclude-patterns "*/test/*,*_test.py"`

**--max-chunk-size SIZE**

- Maximum chunk size in characters
- Default: varies by strategy (typically 1000-2000)
- Example: `--max-chunk-size 1500`

**--min-chunk-size SIZE**

- Minimum chunk size in characters
- Default: 100
- Chunks smaller than this are merged
- Example: `--min-chunk-size 200`

**--overlap SIZE**

- Overlap between chunks (for sliding_window strategy)
- Default: 200
- Example: `--overlap 300`

**--recursive / --no-recursive**

- Whether to search directories recursively
- Default: `--recursive`
- Example: `--no-recursive`

**--verbose / -v**

- Enable verbose output
- Shows progress and statistics
- Example: `--verbose`

### Search Command

```bash
sw-search search [INDEX_PATH] [QUERY] [OPTIONS]
```

**Arguments:**

- `INDEX_PATH` - Path to .swsearch file
- `QUERY` - Search query string

**Options:**

**--backend BACKEND**

- Backend to search
- Options: `sqlite`, `pgvector`
- Default: detected from INDEX_PATH
- Example: `--backend pgvector`

**--connection-string STRING**

- PostgreSQL connection string (for pgvector)
- Example: `--connection-string "postgresql://..."`

**--collection-name NAME**

- Collection to search (for pgvector)
- Example: `--collection-name docs`

**--model MODEL**

- Model to use for query embedding
- Options: `mini`, `base`, `large`
- Must match model used during indexing
- Example: `--model mini`

**--count N**

- Number of results to return
- Default: 5
- Example: `--count 10`

**--threshold FLOAT**

- Minimum similarity threshold (0.0 to 1.0)
- Default: 0.5
- Example: `--threshold 0.4`

**--tags TAG1,TAG2,...**

- Filter by tags
- Example: `--tags api,python`

**--verbose**

- Show similarity scores and metadata
- Example: `--verbose`

### Validate Command

```bash
sw-search validate [INDEX_PATH]
```

**Arguments:**

- `INDEX_PATH` - Path to .swsearch file to validate

Checks index integrity and displays statistics.

### Export Command

```bash
sw-search export [INDEX_PATH] [OUTPUT_PATH]
```

**Arguments:**

- `INDEX_PATH` - Path to .swsearch file
- `OUTPUT_PATH` - Path to output JSON file

**Options:**

**--format FORMAT**

- Export format
- Options: `json`, `jsonl`
- Default: `json`
- Example: `--format jsonl`

### Migrate Command

```bash
sw-search migrate [SOURCE] [OPTIONS]
```

**Arguments:**

- `SOURCE` - Source index (.swsearch file or pgvector collection)

**Options:**

**--from-backend BACKEND**

- Source backend
- Options: `sqlite`, `pgvector`
- Example: `--from-backend sqlite`

**--to-backend BACKEND**

- Destination backend
- Options: `sqlite`, `pgvector`
- Example: `--to-backend pgvector`

**--from-connection-string STRING**

- Source pgvector connection string
- Example: `--from-connection-string "postgresql://..."`

**--to-connection-string STRING**

- Destination pgvector connection string
- Example: `--to-connection-string "postgresql://..."`

**--from-collection-name NAME**

- Source collection name
- Example: `--from-collection-name old_docs`

**--to-collection-name NAME**

- Destination collection name
- Example: `--to-collection-name new_docs`

**--output PATH**

- Output .swsearch file (when migrating to sqlite)
- Example: `--output migrated.swsearch`

### Remote Command

```bash
sw-search remote [ENDPOINT] [QUERY] [OPTIONS]
```

**Arguments:**

- `ENDPOINT` - Search server URL
- `QUERY` - Search query

**Options:**

**--index-name NAME**

- Index to search on remote server
- Example: `--index-name docs`

**--count N**

- Number of results
- Default: 5
- Example: `--count 3`

**--threshold FLOAT**

- Similarity threshold
- Default: 0.5
- Example: `--threshold 0.4`

## native_vector_search Skill Configuration

### Required Parameters

```python
{
    "tool_name": str,           # SWAIG function name
    "description": str,         # When LLM should use this tool
}
```

### Backend Selection (Choose One)

**Option 1: Local .swsearch file**
```python
{
    "index_path": str           # Path to .swsearch file
}
```

**Option 2: Remote search server**
```python
{
    "remote_url": str,          # Search server URL
    "index_name": str           # Index name on server
}
```

**Option 3: pgvector database**
```python
{
    "backend": "pgvector",
    "connection_string": str,   # PostgreSQL connection string
    "collection_name": str,     # Collection name
    "model_name": str           # Model: mini, base, or large
}
```

### Search Behavior Parameters

```python
{
    "count": int,                          # Number of results (default: 5)
    "distance_threshold": float,           # Similarity threshold 0.0-1.0 (default: 0.5)
    "tags": List[str],                     # Filter by tags (optional)
    "max_content_length": int,             # Max total response chars (default: 32768)
}
```

### User Experience Parameters

```python
{
    "no_results_message": str,             # Message when no results (optional)
                                           # Use {query} placeholder

    "response_format_callback": Callable,  # Custom formatter (optional)
                                           # Signature: (response, agent, query, results, **kwargs) -> str

    "swaig_fields": dict,                  # SWAIG configuration (optional)
}
```

### SWAIG Fields Structure

```python
{
    "swaig_fields": {
        "fillers": {
            "en-US": [                     # Language-specific fillers
                "Searching...",
                "Let me look that up...",
                "Checking the documentation..."
            ]
        },
        "wait_file": str,                  # Audio file URL for hold music (optional)
        "wait_file_loops": int             # Number of loops (optional)
    }
}
```

### Complete Example

```python
self.add_skill("native_vector_search", {
    # Identity
    "tool_name": "search_product_docs",
    "description": "Search comprehensive product documentation including features, APIs, configuration, and troubleshooting",

    # Backend (pgvector)
    "backend": "pgvector",
    "connection_string": "postgresql://user:pass@localhost:5432/knowledge",
    "collection_name": "product_docs",
    "model_name": "mini",

    # Search behavior
    "count": 5,
    "distance_threshold": 0.4,
    "tags": ["documentation", "api"],
    "max_content_length": 32768,

    # User experience
    "no_results_message": "I couldn't find information about '{query}' in our documentation. Please rephrase or try a different question.",
    "response_format_callback": self._custom_formatter,

    # SWAIG configuration
    "swaig_fields": {
        "fillers": {
            "en-US": [
                "Let me search the documentation...",
                "I'm looking through our knowledge base...",
                "Searching for that information...",
                "Let me check the docs..."
            ]
        }
    }
})
```

## JSON Schema for Pre-chunked Content

When using the JSON workflow with `--chunking-strategy json`:

```json
{
  "chunks": [
    {
      "content": "string (required)",
      "metadata": {
        "filename": "string (optional)",
        "section": "string (optional)",
        "h1": "string (optional)",
        "h2": "string (optional)",
        "h3": "string (optional)",
        "tags": ["string", "string", ...] (optional),
        "category": "string (optional)",
        "priority": "string (optional)",
        "language": "string (optional)",
        "difficulty": "string (optional)",
        "has_code": boolean (optional),
        "code_languages": ["string", ...] (optional),
        "custom_field": "any_type (optional)"
      }
    }
  ]
}
```

**Notes:**

- Only `content` is required
- All metadata fields are optional
- You can add custom metadata fields
- `tags` array is used for filtering and hybrid search
- Standard metadata fields: `filename`, `section`, `h1`-`h6`, `tags`, `category`, `has_code`, `code_languages`

## Environment Variables

### Search System

**TRANSFORMERS_CACHE**

- Directory for cached models
- Default: `~/.cache/huggingface/transformers`
- Example: `export TRANSFORMERS_CACHE=/data/models`

**TRANSFORMERS_OFFLINE**

- Use offline mode (don't download models)
- Set to `1` to enable
- Example: `export TRANSFORMERS_OFFLINE=1`

**SEARCH_DEBUG**

- Enable debug logging for search operations
- Set to `1` to enable
- Example: `export SEARCH_DEBUG=1`

### pgvector Configuration

**PGVECTOR_CONNECTION**

- Default PostgreSQL connection string
- Format: `postgresql://user:pass@host:port/database`
- Example: `export PGVECTOR_CONNECTION="postgresql://user:pass@localhost:5432/knowledge"`

**PGVECTOR_DB_USER**

- PostgreSQL username
- Example: `export PGVECTOR_DB_USER=signalwire`

**PGVECTOR_DB_PASSWORD**

- PostgreSQL password
- Example: `export PGVECTOR_DB_PASSWORD=secure_password`

**PGVECTOR_HOST**

- PostgreSQL host
- Default: `localhost`
- Example: `export PGVECTOR_HOST=postgres.example.com`

**PGVECTOR_PORT**

- PostgreSQL port
- Default: `5432`
- Example: `export PGVECTOR_PORT=5432`

**PGVECTOR_DB_NAME**

- PostgreSQL database name
- Example: `export PGVECTOR_DB_NAME=knowledge`

## PostgreSQL Configuration

### pgvector Extension Setup

```sql
-- Install extension (as superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for collection
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id SERIAL PRIMARY KEY,
    collection_name TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384 for mini, 768 for base, 1024 for large
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

CREATE INDEX ON knowledge_chunks (collection_name);

CREATE INDEX ON knowledge_chunks USING gin (metadata jsonb_path_ops);
```

### Recommended PostgreSQL Settings

```ini
# postgresql.conf

# Memory settings
shared_buffers = 4GB                    # 25% of system RAM
effective_cache_size = 12GB             # 75% of system RAM
work_mem = 128MB                        # For sorting/aggregation
maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX

# Connection settings
max_connections = 100                   # Adjust based on load

# Performance settings
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD storage

# Logging
log_min_duration_statement = 1000       # Log slow queries (>1s)
```

## Distance Threshold Guidelines

| Content Type | Recommended Threshold | Notes |
|--------------|----------------------|-------|
| Technical documentation | 0.4 | Balanced precision/recall |
| FAQ content | 0.5 | Moderate matching |
| Creative content | 0.6 | Broader matching |
| Exact lookups | 0.3 | Strict matching |
| Code examples | 0.4 | With markdown strategy |
| General knowledge | 0.5 | Permissive |

## Result Count Guidelines

| Use Case | Recommended Count | Notes |
|----------|------------------|-------|
| Specific questions | 3 | Focused answer |
| Exploratory questions | 5 | Balanced coverage |
| Research queries | 7-10 | Comprehensive |
| Voice agents | 3 | Concise for speech |
| Chat agents | 5 | Detailed for text |

## Max Content Length Guidelines

| Agent Type | Recommended Length | Notes |
|------------|-------------------|-------|
| Voice agents | 16384 (16KB) | Faster, focused |
| Chat agents | 32768 (32KB) | Default, balanced |
| Complex queries | 65536 (64KB) | Comprehensive |
| Serverless | 16384 (16KB) | Minimize size |

## Model Comparison

| Model | Dimensions | Speed | Quality | Size | Use Case |
|-------|-----------|-------|---------|------|----------|
| mini | 384 | Fast (1000 chunks/s) | Good | ~100MB | Production default |
| base | 768 | Medium (500 chunks/s) | Better | ~400MB | Quality matters |
| large | 1024 | Slow (200 chunks/s) | Best | ~1.4GB | Research/special |

## Chunking Strategy Quick Reference

| Strategy | Best For | Chunk Size | Pros | Cons |
|----------|----------|------------|------|------|
| sentence | General text | Small | Fast, simple | May split context |
| paragraph | Articles, blogs | Medium | Natural boundaries | Variable size |
| page | Books, papers | Large | Preserves context | May be too large |
| sliding_window | Dense content | Medium | Overlapping context | Duplication |
| semantic | Narrative text | Variable | Topic coherence | Slower |
| topic | Long documents | Variable | Topic-based | Requires NLP |
| qa | FAQ content | Medium | Keeps Q&A together | Needs structure |
| markdown | Technical docs | Variable | Preserves code | Markdown required |
| json | Pre-chunked | Any | Full control | Manual work |

## Performance Benchmarks

### Index Build Times (1,000 documents, ~20,000 chunks)

| Configuration | Build Time | Index Size |
|--------------|------------|------------|
| mini + sentence | 3 minutes | 30MB |
| mini + markdown | 8 minutes | 80MB |
| base + sentence | 7 minutes | 120MB |
| base + markdown | 15 minutes | 160MB |

### Query Performance (per query)

| Configuration | Latency | Notes |
|--------------|---------|-------|
| SQLite + mini | 15-25ms | Local file |
| SQLite + base | 25-40ms | Local file |
| pgvector + mini | 20-35ms | With indexes |
| pgvector + base | 35-50ms | With indexes |

### Memory Usage

| Configuration | Runtime Memory |
|--------------|----------------|
| mini model | ~1.5GB |
| base model | ~2.5GB |
| large model | ~4GB |
| query-only mode | -400MB |

This reference covers all configuration options available in the SignalWire Agents SDK search system.
# Appendix B: JSON Schema for Pre-chunked Content

This appendix provides the complete JSON schema for the manual chunking workflow, along with examples and validation rules.

## Complete JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["chunks"],
  "properties": {
    "chunks": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["content"],
        "properties": {
          "content": {
            "type": "string",
            "minLength": 1,
            "description": "The text content of the chunk"
          },
          "metadata": {
            "type": "object",
            "properties": {
              "filename": {
                "type": "string",
                "description": "Source filename"
              },
              "section": {
                "type": "string",
                "description": "Section or chapter name"
              },
              "h1": {
                "type": "string",
                "description": "Top-level heading"
              },
              "h2": {
                "type": "string",
                "description": "Second-level heading"
              },
              "h3": {
                "type": "string",
                "description": "Third-level heading"
              },
              "h4": {
                "type": "string",
                "description": "Fourth-level heading"
              },
              "h5": {
                "type": "string",
                "description": "Fifth-level heading"
              },
              "h6": {
                "type": "string",
                "description": "Sixth-level heading"
              },
              "tags": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "Tags for filtering and hybrid search"
              },
              "category": {
                "type": "string",
                "description": "Content category"
              },
              "subcategory": {
                "type": "string",
                "description": "Content subcategory"
              },
              "priority": {
                "type": "string",
                "enum": ["critical", "high", "normal", "low"],
                "description": "Content priority"
              },
              "difficulty": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced", "expert"],
                "description": "Content difficulty level"
              },
              "audience": {
                "type": "string",
                "description": "Target audience"
              },
              "language": {
                "type": "string",
                "pattern": "^[a-z]{2}(-[A-Z]{2})?$",
                "description": "Language code (e.g., en, en-US, fr)"
              },
              "has_code": {
                "type": "boolean",
                "description": "Whether chunk contains code"
              },
              "code_languages": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "Programming languages in chunk"
              },
              "has_url": {
                "type": "boolean",
                "description": "Whether chunk contains URLs"
              },
              "urls": {
                "type": "array",
                "items": {
                  "type": "string",
                  "format": "uri"
                },
                "description": "URLs mentioned in chunk"
              },
              "version": {
                "type": "string",
                "description": "Product version"
              },
              "created_at": {
                "type": "string",
                "format": "date-time",
                "description": "Creation timestamp"
              },
              "updated_at": {
                "type": "string",
                "format": "date-time",
                "description": "Last update timestamp"
              },
              "author": {
                "type": "string",
                "description": "Content author"
              },
              "source": {
                "type": "string",
                "description": "Content source"
              },
              "verified": {
                "type": "boolean",
                "description": "Whether content is verified"
              },
              "chunk_id": {
                "type": "string",
                "description": "Unique chunk identifier"
              },
              "related_chunks": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "IDs of related chunks"
              },
              "prerequisite": {
                "type": "string",
                "description": "Prerequisite chunk ID"
              },
              "next_topic": {
                "type": "string",
                "description": "Next topic chunk ID"
              }
            },
            "additionalProperties": true
          }
        }
      }
    }
  }
}
```

## Minimal Example

The simplest valid JSON:

```json
{
  "chunks": [
    {
      "content": "This is a chunk of text."
    }
  ]
}
```

## Standard Example

Typical documentation chunk:

```json
{
  "chunks": [
    {
      "content": "To create an agent, inherit from AgentBase and implement your methods...",
      "metadata": {
        "filename": "getting_started.md",
        "section": "Creating Agents",
        "h1": "Getting Started",
        "h2": "Creating Your First Agent",
        "tags": ["agent", "getting-started", "tutorial"],
        "category": "documentation",
        "difficulty": "beginner",
        "has_code": true,
        "code_languages": ["python"]
      }
    }
  ]
}
```

## Complete Example

Fully populated chunk with all common fields:

```json
{
  "chunks": [
    {
      "content": "Here's how to authenticate with the API:\n\n```python\nfrom signalwire_agents import AgentBase\n\nagent = AgentBase(\n    name=\"MyAgent\",\n    api_key=\"your_key\"\n)\n```\n\nThis creates an authenticated agent instance.",
      "metadata": {
        "chunk_id": "auth_example_001",
        "filename": "authentication.md",
        "section": "API Authentication",
        "h1": "Authentication",
        "h2": "Getting Started",
        "h3": "Python Example",
        "tags": [
          "authentication",
          "api",
          "python",
          "code",
          "example",
          "getting-started"
        ],
        "category": "documentation",
        "subcategory": "authentication",
        "priority": "high",
        "difficulty": "beginner",
        "audience": "developers",
        "language": "en-US",
        "has_code": true,
        "code_languages": ["python"],
        "has_url": false,
        "version": "2.0",
        "created_at": "2025-01-15T10:00:00Z",
        "updated_at": "2025-01-20T15:30:00Z",
        "author": "docs-team",
        "source": "official-docs",
        "verified": true,
        "related_chunks": ["auth_example_002", "auth_troubleshooting"],
        "next_topic": "authorization"
      }
    }
  ]
}
```

## Multi-Chunk Example

Multiple chunks in one file:

```json
{
  "chunks": [
    {
      "content": "Authentication is the process of verifying identity...",
      "metadata": {
        "chunk_id": "auth_001",
        "h1": "Authentication",
        "h2": "Overview",
        "tags": ["authentication", "security", "overview"],
        "difficulty": "beginner"
      }
    },
    {
      "content": "To authenticate, use Bearer tokens...",
      "metadata": {
        "chunk_id": "auth_002",
        "h1": "Authentication",
        "h2": "Bearer Tokens",
        "tags": ["authentication", "security", "bearer-token"],
        "difficulty": "beginner",
        "prerequisite": "auth_001"
      }
    },
    {
      "content": "Here's a Python example of authentication...",
      "metadata": {
        "chunk_id": "auth_003",
        "h1": "Authentication",
        "h2": "Examples",
        "h3": "Python",
        "tags": ["authentication", "python", "code", "example"],
        "difficulty": "intermediate",
        "has_code": true,
        "code_languages": ["python"],
        "prerequisite": "auth_002"
      }
    }
  ]
}
```

## Field Descriptions

### Required Fields

**content** (string, required)
- The actual text content of the chunk
- Minimum length: 1 character
- Can include markdown formatting
- Code blocks are preserved

### Standard Metadata Fields

**filename** (string)
- Original source file name
- Example: `"getting_started.md"`

**section** (string)
- Logical section or chapter name
- Example: `"API Reference"`

**h1 through h6** (string)
- Heading hierarchy from source document
- Provides context for the chunk
- Example: `"h1": "Getting Started", "h2": "Installation"`

**tags** (array of strings)
- Keywords for filtering and hybrid search
- Used to boost relevance when matched
- Example: `["authentication", "security", "api"]`

**category** (string)
- High-level content category
- Example: `"documentation"`, `"tutorial"`, `"reference"`

**subcategory** (string)
- More specific categorization
- Example: `"authentication"`, `"configuration"`

**priority** (string enum)
- Content importance level
- Values: `"critical"`, `"high"`, `"normal"`, `"low"`
- Example: `"high"`

**difficulty** (string enum)
- Content complexity level
- Values: `"beginner"`, `"intermediate"`, `"advanced"`, `"expert"`
- Example: `"beginner"`

**audience** (string)
- Target audience description
- Example: `"developers"`, `"administrators"`, `"end-users"`

**language** (string)
- ISO language code
- Format: `xx` or `xx-YY`
- Example: `"en"`, `"en-US"`, `"fr-FR"`

**has_code** (boolean)
- Whether chunk contains code examples
- Automatically detected by markdown strategy
- Used for hybrid search boosting (+20%)
- Example: `true`

**code_languages** (array of strings)
- Programming languages present in chunk
- Example: `["python", "javascript"]`

**has_url** (boolean)
- Whether chunk contains URLs
- Example: `true`

**urls** (array of strings)
- Actual URLs mentioned in chunk
- Must be valid URIs
- Example: `["https://example.com/docs"]`

**version** (string)
- Product version this content applies to
- Example: `"2.0"`, `"1.5.3"`

**created_at** (string, ISO 8601)
- When chunk was created
- Format: `YYYY-MM-DDTHH:MM:SSZ`
- Example: `"2025-01-15T10:00:00Z"`

**updated_at** (string, ISO 8601)
- When chunk was last updated
- Format: `YYYY-MM-DDTHH:MM:SSZ`
- Example: `"2025-01-20T15:30:00Z"`

**author** (string)
- Content author or team
- Example: `"docs-team"`, `"john.doe@example.com"`

**source** (string)
- Content source identifier
- Example: `"official-docs"`, `"blog"`, `"support-kb"`

**verified** (boolean)
- Whether content has been verified/reviewed
- Example: `true`

### Relationship Fields

**chunk_id** (string)
- Unique identifier for this chunk
- Used for linking chunks together
- Example: `"auth_001"`

**related_chunks** (array of strings)
- IDs of related chunks
- Enables building related content lists
- Example: `["auth_002", "auth_troubleshooting"]`

**prerequisite** (string)
- Chunk ID that should be read first
- Enables building learning paths
- Example: `"installation_001"`

**next_topic** (string)
- Chunk ID for next topic in sequence
- Enables sequential navigation
- Example: `"authorization_001"`

### Custom Fields

You can add any custom fields to metadata:

```json
{
  "metadata": {
    "tags": ["example"],
    "custom_field": "custom_value",
    "estimated_time_minutes": 5,
    "prerequisites": ["topic_a", "topic_b"],
    "internal_id": "DOC-12345",
    "review_status": "approved",
    "seo_keywords": ["keyword1", "keyword2"]
  }
}
```

All custom fields are stored and preserved but not used by search unless you implement custom logic.

## Building Index from JSON

```bash
sw-search ./chunks.json --chunking-strategy json --output knowledge.swsearch
```

The `json` strategy:
- Reads `chunks` array
- Preserves all metadata exactly
- Skips text processing (already chunked)
- Generates embeddings for `content` field
- Stores chunks with metadata intact

## Validation

### Using JSON Schema Validator

```python
import json
import jsonschema

# Load schema
with open('chunk_schema.json') as f:
    schema = json.load(f)

# Load your chunks
with open('chunks.json') as f:
    chunks = json.load(f)

# Validate
try:
    jsonschema.validate(chunks, schema)
    print("✓ Valid JSON")
except jsonschema.ValidationError as e:
    print(f"✗ Invalid: {e.message}")
```

### Common Validation Errors

**Missing required field:**
```json
{
  "chunks": [
    {
      "metadata": {"tags": ["example"]}
      // Error: missing "content" field
    }
  ]
}
```

**Empty content:**
```json
{
  "chunks": [
    {
      "content": ""  // Error: content must be non-empty
    }
  ]
}
```

**Invalid priority:**
```json
{
  "chunks": [
    {
      "content": "...",
      "metadata": {
        "priority": "super-high"  // Error: must be critical/high/normal/low
      }
    }
  ]
}
```

**Invalid language code:**
```json
{
  "chunks": [
    {
      "content": "...",
      "metadata": {
        "language": "english"  // Error: must be ISO code like "en"
      }
    }
  ]
}
```

## Programmatic Generation

### Python Example

```python
import json
from datetime import datetime
from typing import List, Dict, Any

class ChunkBuilder:
    """Build JSON chunks programmatically"""

    def __init__(self):
        self.chunks = []

    def add_chunk(
        self,
        content: str,
        tags: List[str] = None,
        category: str = None,
        difficulty: str = None,
        has_code: bool = False,
        **extra_metadata
    ):
        """Add a chunk with metadata"""
        chunk = {
            "content": content,
            "metadata": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                **extra_metadata
            }
        }

        if tags:
            chunk["metadata"]["tags"] = tags
        if category:
            chunk["metadata"]["category"] = category
        if difficulty:
            chunk["metadata"]["difficulty"] = difficulty
        if has_code:
            chunk["metadata"]["has_code"] = has_code

        self.chunks.append(chunk)

    def to_json(self, output_path: str):
        """Write chunks to JSON file"""
        with open(output_path, 'w') as f:
            json.dump({"chunks": self.chunks}, f, indent=2)

# Usage
builder = ChunkBuilder()

builder.add_chunk(
    content="Getting started with the API...",
    tags=["getting-started", "api"],
    category="documentation",
    difficulty="beginner",
    h1="Getting Started",
    h2="Overview"
)

builder.add_chunk(
    content="Here's a Python example:\n```python\ncode here\n```",
    tags=["python", "code", "example"],
    category="documentation",
    difficulty="intermediate",
    has_code=True,
    code_languages=["python"],
    h1="Getting Started",
    h2="Examples"
)

builder.to_json("chunks.json")
```

### JavaScript Example

```javascript
const fs = require('fs');

class ChunkBuilder {
  constructor() {
    this.chunks = [];
  }

  addChunk(content, metadata = {}) {
    this.chunks.push({
      content,
      metadata: {
        created_at: new Date().toISOString(),
        ...metadata
      }
    });
  }

  toJSON(outputPath) {
    const data = JSON.stringify(
      { chunks: this.chunks },
      null,
      2
    );
    fs.writeFileSync(outputPath, data);
  }
}

// Usage
const builder = new ChunkBuilder();

builder.addChunk(
  "Getting started with the API...",
  {
    tags: ["getting-started", "api"],
    category: "documentation",
    difficulty: "beginner",
    h1: "Getting Started",
    h2: "Overview"
  }
);

builder.addChunk(
  "Here's a JavaScript example:\n```javascript\ncode here\n```",
  {
    tags: ["javascript", "code", "example"],
    category: "documentation",
    difficulty: "intermediate",
    has_code: true,
    code_languages: ["javascript"],
    h1: "Getting Started",
    h2: "Examples"
  }
);

builder.toJSON("chunks.json");
```

## Best Practices

### 1. Always Include Tags

```json
{
  "content": "...",
  "metadata": {
    "tags": ["relevant", "keywords", "for", "search"]
  }
}
```

Tags are critical for hybrid search boosting.

### 2. Use Hierarchical Metadata

```json
{
  "metadata": {
    "h1": "Top Section",
    "h2": "Subsection",
    "h3": "Specific Topic"
  }
}
```

Provides context for where chunk came from.

### 3. Mark Code Chunks

```json
{
  "content": "Code example...",
  "metadata": {
    "has_code": true,
    "code_languages": ["python"],
    "tags": ["code", "example"]
  }
}
```

Gets +20% boost in hybrid search.

### 4. Use Consistent Categories

```json
// Good: consistent categories
{"category": "documentation"}
{"category": "documentation"}

// Bad: inconsistent
{"category": "documentation"}
{"category": "docs"}
```

### 5. Link Related Chunks

```json
{
  "metadata": {
    "chunk_id": "auth_001",
    "next_topic": "auth_002",
    "related_chunks": ["security_001", "api_keys_001"]
  }
}
```

Enables building navigation and recommendations.

### 6. Include Timestamps

```json
{
  "metadata": {
    "created_at": "2025-01-15T10:00:00Z",
    "updated_at": "2025-01-20T15:30:00Z"
  }
}
```

Track content freshness.

### 7. Validate Before Building

```bash
# Validate JSON syntax
python -m json.tool chunks.json > /dev/null && echo "Valid JSON"

# Validate against schema
jsonschema -i chunks.json chunk_schema.json
```

## Converting Markdown to JSON

Example script to extract metadata from markdown:

```python
import re
import json
from pathlib import Path

def extract_frontmatter(content):
    """Extract YAML frontmatter from markdown"""
    match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if match:
        import yaml
        return yaml.safe_load(match.group(1))
    return {}

def markdown_to_chunks(md_path):
    """Convert markdown file to chunks JSON"""
    content = Path(md_path).read_text()
    frontmatter = extract_frontmatter(content)

    # Split by headers
    sections = re.split(r'^(#{1,6})\s+(.+)$', content, flags=re.MULTILINE)

    chunks = []
    current_h1 = None
    current_h2 = None

    for i in range(1, len(sections), 3):
        if i+2 >= len(sections):
            break

        level = len(sections[i])
        heading = sections[i+1]
        text = sections[i+2].strip()

        if level == 1:
            current_h1 = heading
            current_h2 = None
        elif level == 2:
            current_h2 = heading

        if text:
            chunk = {
                "content": text,
                "metadata": {
                    "filename": Path(md_path).name,
                    "h1": current_h1,
                    "h2": current_h2,
                    "has_code": "```" in text,
                    **frontmatter
                }
            }
            chunks.append(chunk)

    return {"chunks": chunks}

# Usage
result = markdown_to_chunks("docs/getting_started.md")
with open("chunks.json", "w") as f:
    json.dump(result, f, indent=2)
```

This JSON schema and examples provide complete control over chunk creation for the manual workflow.
# Appendix C: Embedding Model Details

This appendix provides detailed technical information about the embedding models used in the SignalWire Agents SDK search system.

## Available Models

The SDK supports three models from the sentence-transformers library:

| Model | Identifier | Dimensions | Size | Speed |
|-------|-----------|------------|------|-------|
| Mini | `all-MiniLM-L6-v2` | 384 | ~90MB | Fast |
| Base | `all-mpnet-base-v2` | 768 | ~420MB | Medium |
| Large | `all-roberta-large-v1` | 1024 | ~1.4GB | Slow |

## Mini Model (all-MiniLM-L6-v2)

### Overview

The default and recommended model for most use cases.

**Full name:** `sentence-transformers/all-MiniLM-L6-v2`

**Architecture:**

- Based on Microsoft's MiniLM
- 6-layer transformer
- 384-dimensional embeddings
- 22.7 million parameters

**Performance:**

- Embedding speed: ~1,000 chunks/second (CPU)
- Embedding speed: ~5,000 chunks/second (GPU)
- Query embedding: 5-8ms (CPU)
- Model load time: 1-2 seconds

**Memory:**

- Model size: ~90MB on disk
- Runtime memory: ~1GB total (with overhead)

### Training Data

Trained on over 1 billion sentence pairs from:
- Natural language inference datasets
- Semantic textual similarity datasets
- Paraphrase datasets
- Question-answer pairs
- Multiple languages (English-focused)

### Benchmark Performance

**Semantic Textual Similarity (STS):**

- STS Benchmark: 82.41
- SICK-R: 78.23

**Information Retrieval:**

- MS MARCO (MRR@10): 32.3
- TREC-COVID (NDCG@10): 63.2

**Clustering:**

- Average: 40.2

**Quality vs Speed:**

- 95% of base model quality
- 2-3x faster than base
- Best balance for production

### Best For

- Production deployments
- Voice agents (latency matters)
- Large knowledge bases (>50K chunks)
- Cost-sensitive deployments
- General documentation search
- FAQ systems
- Real-time applications

### Example Usage

```python
# In agent
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search documentation",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "docs",
    "model_name": "mini"  # Fast, efficient
})
```

```bash
# Building index
sw-search ./docs --model mini --output docs.swsearch
```

## Base Model (all-mpnet-base-v2)

### Overview

Higher quality embeddings with moderate performance impact.

**Full name:** `sentence-transformers/all-mpnet-base-v2`

**Architecture:**

- Based on Microsoft's MPNet
- 12-layer transformer
- 768-dimensional embeddings
- 109 million parameters

**Performance:**

- Embedding speed: ~500 chunks/second (CPU)
- Embedding speed: ~2,500 chunks/second (GPU)
- Query embedding: 10-15ms (CPU)
- Model load time: 2-3 seconds

**Memory:**

- Model size: ~420MB on disk
- Runtime memory: ~2.5GB total (with overhead)

### Training Data

Trained on over 1 billion sentence pairs:
- Same datasets as mini
- Additional domain-specific data
- Longer context windows
- More diverse examples

### Benchmark Performance

**Semantic Textual Similarity (STS):**

- STS Benchmark: 86.99
- SICK-R: 84.57

**Information Retrieval:**

- MS MARCO (MRR@10): 35.8
- TREC-COVID (NDCG@10): 69.4

**Clustering:**

- Average: 44.5

**Quality vs Speed:**

- Reference quality (100%)
- 2x slower than mini
- 5% better accuracy than mini

### Best For

- Quality-critical applications
- Complex semantic searches
- Research and analysis
- Legal/medical documentation
- Academic papers
- Nuanced language understanding
- Multi-lingual content (secondary)

### Example Usage

```python
# In agent - quality matters
self.add_skill("native_vector_search", {
    "tool_name": "search_legal_docs",
    "description": "Search legal documentation",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "legal",
    "model_name": "base"  # Better quality
})
```

```bash
# Building index
sw-search ./legal-docs --model base --output legal.swsearch
```

## Large Model (all-roberta-large-v1)

### Overview

Highest quality embeddings with significant resource requirements.

**Full name:** `sentence-transformers/all-roberta-large-v1`

**Architecture:**

- Based on Facebook's RoBERTa
- 24-layer transformer
- 1024-dimensional embeddings
- 355 million parameters

**Performance:**

- Embedding speed: ~200 chunks/second (CPU)
- Embedding speed: ~1,000 chunks/second (GPU)
- Query embedding: 20-30ms (CPU)
- Model load time: 4-6 seconds

**Memory:**

- Model size: ~1.4GB on disk
- Runtime memory: ~4GB total (with overhead)

### Training Data

Trained on:
- 1 billion+ sentence pairs
- Extended context examples
- Domain-specific corpora
- Multi-task learning objectives

### Benchmark Performance

**Semantic Textual Similarity (STS):**

- STS Benchmark: 88.45
- SICK-R: 86.32

**Information Retrieval:**

- MS MARCO (MRR@10): 37.2
- TREC-COVID (NDCG@10): 71.8

**Clustering:**

- Average: 47.1

**Quality vs Speed:**

- Highest quality (+2% over base)
- 5x slower than mini
- 2.5x slower than base

### Best For

- Research projects
- Specialized domains
- Very nuanced searches
- When accuracy > latency
- Offline batch processing
- Not recommended for production

### Example Usage

```python
# Research use case only
self.add_skill("native_vector_search", {
    "tool_name": "search_research",
    "description": "Search research papers",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "research",
    "model_name": "large"  # Highest quality
})
```

```bash
# Building index (slow)
sw-search ./research --model large --output research.swsearch
```

## Model Comparison

### Accuracy Comparison

Testing on 1,000 technical documentation queries:

| Model | Precision@5 | Recall@5 | MRR | Avg Similarity |
|-------|-------------|----------|-----|----------------|
| Mini | 0.847 | 0.923 | 0.782 | 0.654 |
| Base | 0.891 | 0.951 | 0.823 | 0.687 |
| Large | 0.903 | 0.959 | 0.841 | 0.701 |

**Key insight:** Mini achieves 94% of base model accuracy with 2-3x speed improvement.

### Speed Comparison

Building 10,000 chunks on CPU (Intel i7-10700K):

| Model | Build Time | Chunks/Second | Index Size |
|-------|------------|---------------|------------|
| Mini | 10 minutes | 1,000 | 40MB |
| Base | 20 minutes | 500 | 80MB |
| Large | 50 minutes | 200 | 120MB |

### Memory Comparison

Runtime memory usage (querying):

| Model | Model Size | Peak Memory | Concurrent Agents |
|-------|-----------|-------------|-------------------|
| Mini | 90MB | 1.5GB | 16 per 32GB |
| Base | 420MB | 2.5GB | 10 per 32GB |
| Large | 1.4GB | 4GB | 6 per 32GB |

### Cost Comparison

Monthly infrastructure costs for 1 million queries:

| Model | Compute Cost | Memory Cost | Total |
|-------|-------------|-------------|-------|
| Mini | $50 | $20 | $70 |
| Base | $100 | $40 | $140 |
| Large | $250 | $80 | $330 |

Estimates based on cloud compute pricing.

## Technical Details

### Embedding Generation

All models use the sentence-transformers library:

```python
from sentence_transformers import SentenceTransformer

# Load model (cached after first load)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embedding for text
text = "How do I authenticate with the API?"
embedding = model.encode(text)

# Result: numpy array of shape (384,) for mini
print(embedding.shape)  # (384,)
print(type(embedding))  # <class 'numpy.ndarray'>
```

### Similarity Computation

Cosine similarity between embeddings:

```python
import numpy as np

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Example
query_embedding = model.encode("authentication")
doc_embedding = model.encode("how to authenticate")

similarity = cosine_similarity(query_embedding, doc_embedding)
print(f"Similarity: {similarity:.3f}")  # e.g., 0.847
```

### Dimension Impact

Higher dimensions = more information:

**Mini (384 dims):**

- Compact representation
- Faster operations
- Good for general text

**Base (768 dims):**

- Richer representation
- Captures more nuance
- Better for complex text

**Large (1024 dims):**

- Most detailed representation
- Captures subtle differences
- Best for specialized domains

### Vector Storage

Storage requirements by model:

**SQLite (.swsearch files):**
```
Per chunk storage:
- Mini: 384 floats × 4 bytes = 1,536 bytes
- Base: 768 floats × 4 bytes = 3,072 bytes
- Large: 1024 floats × 4 bytes = 4,096 bytes

For 10,000 chunks:
- Mini: ~15MB (vectors only)
- Base: ~30MB (vectors only)
- Large: ~40MB (vectors only)

Plus content, metadata, indexes (~2-3x)
```

**pgvector:**
```sql
-- Mini model (384 dimensions)
CREATE TABLE chunks (
    embedding vector(384)  -- 1,536 bytes per row
);

-- Base model (768 dimensions)
CREATE TABLE chunks (
    embedding vector(768)  -- 3,072 bytes per row
);

-- Large model (1024 dimensions)
CREATE TABLE chunks (
    embedding vector(1024)  -- 4,096 bytes per row
);
```

## Model Selection Guide

### Decision Tree

```
Start here: Do you need the absolute highest quality?
├─ Yes → Use base model
│  └─ Is latency critical (voice)?
│     ├─ Yes → Use mini (quality loss minimal)
│     └─ No → Stick with base
│
└─ No → Use mini model
   └─ Quality issues in testing?
      ├─ Yes → Upgrade to base
      └─ No → Stick with mini
```

### By Use Case

**Voice Agents:**

- **Recommendation:** Mini
- **Reason:** Latency critical, 5-8ms query time
- **Quality:** Sufficient for conversational queries

**Chat Agents:**

- **Recommendation:** Mini or Base
- **Reason:** Latency less critical
- **Quality:** Base if nuance matters

**FAQ Systems:**

- **Recommendation:** Mini
- **Reason:** Queries are straightforward
- **Quality:** Mini handles direct questions well

**Technical Documentation:**

- **Recommendation:** Mini
- **Reason:** Good with structured text
- **Quality:** Code + markdown strategy helps

**Legal/Medical:**

- **Recommendation:** Base
- **Reason:** Nuance and accuracy critical
- **Quality:** Worth the performance cost

**Research/Academic:**

- **Recommendation:** Base or Large
- **Reason:** Complex language, subtle distinctions
- **Quality:** Highest accuracy needed

**Multi-lingual:**

- **Recommendation:** Base
- **Reason:** Better cross-lingual transfer
- **Quality:** Trained on more languages

## Hardware Recommendations

### For Mini Model

**Development:**

- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB for models/cache

**Production (single agent):**

- CPU: 2-4 cores
- RAM: 4GB
- Storage: 5GB

**Production (multiple agents):**

- CPU: 8+ cores
- RAM: 16GB (1.5GB per agent)
- Storage: 10GB

### For Base Model

**Development:**

- CPU: 8+ cores
- RAM: 16GB
- Storage: 15GB

**Production (single agent):**

- CPU: 4-8 cores
- RAM: 8GB
- Storage: 10GB

**Production (multiple agents):**

- CPU: 16+ cores
- RAM: 32GB (2.5GB per agent)
- Storage: 20GB

### GPU Acceleration

All models support GPU acceleration:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
else:
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
```

**Speed improvement with GPU:**

- Mini: 5x faster
- Base: 5x faster
- Large: 5x faster

**GPU requirements:**

- Mini: 2GB VRAM
- Base: 4GB VRAM
- Large: 8GB VRAM

## Model Updates and Versioning

### Version Pinning

Models are versioned:

```python
# Specific version (recommended for production)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Latest (not recommended for production)
# May change between builds
```

### Updating Models

**To update to newer version:**

1. Clear cache:
```bash
rm -rf ~/.cache/huggingface/transformers/
```

2. Rebuild indexes:
```bash
sw-search ./docs --model mini --output docs_v2.swsearch
```

3. Update agent configuration:
```python
{
    "model_name": "mini",
    "index_path": "./docs_v2.swsearch"
}
```

### Model Compatibility

**Important:** Indexes built with one model cannot be queried with another:

```bash
# ❌ WRONG: Build with base, query with mini
sw-search ./docs --model base --output docs.swsearch

# Agent uses mini
{"model_name": "mini"}  # Will give incorrect results!

# ✅ CORRECT: Match models
sw-search ./docs --model base --output docs.swsearch
{"model_name": "base"}  # Correct
```

## Custom Models

The system uses sentence-transformers, which supports custom models:

```python
# Use custom model from Hugging Face
model = SentenceTransformer('your-org/your-model')
```

**Requirements for custom models:**

- Must be compatible with sentence-transformers
- Must output fixed-size embeddings
- Must support `.encode()` method

**Not currently exposed via CLI**, but possible via Python API.

## Key Takeaways

1. **Mini model recommended** - 94% of base quality, 2-3x faster
2. **Base for quality-critical** - legal, medical, research
3. **Large rarely needed** - marginal improvement, high cost
4. **Models must match** - build and query with same model
5. **GPU 5x faster** - if available, use it for building
6. **Memory scales with model** - mini: 1.5GB, base: 2.5GB, large: 4GB
7. **Dimensions matter** - higher = more nuance but slower
8. **Benchmark carefully** - test with your actual queries

For 95% of use cases, mini model is the right choice.
# Appendix D: Migration Guide

This appendix provides step-by-step guides for common migration scenarios when working with the SignalWire Agents SDK search system.

## Migration Scenarios

1. [SQLite to pgvector](#sqlite-to-pgvector)
2. [pgvector to SQLite](#pgvector-to-sqlite)
3. [Between pgvector collections](#between-pgvector-collections)
4. [Changing embedding models](#changing-embedding-models)
5. [Changing chunking strategies](#changing-chunking-strategies)
6. [From other vector databases](#from-other-vector-databases)
7. [Updating production indexes](#updating-production-indexes)

---

## SQLite to pgvector

**When to migrate:**

- Scaling to multiple agents
- Need concurrent access
- Want shared knowledge base
- Scaling beyond 100K chunks

### Step 1: Setup pgvector

```bash
# Install PostgreSQL with pgvector
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-15-pgvector

# macOS
brew install postgresql pgvector

# Start PostgreSQL
sudo systemctl start postgresql
```

```sql
-- Create database
CREATE DATABASE knowledge;

-- Connect
\c knowledge

-- Install extension
CREATE EXTENSION vector;
```

### Step 2: Verify Existing Index

```bash
# Check current index
sw-search validate ./knowledge.swsearch

# Test a query
sw-search search ./knowledge.swsearch "test query" --verbose
```

Note the model used (mini/base/large) - you'll need to match it.

### Step 3: Export from SQLite

```bash
# Export to JSON
sw-search export ./knowledge.swsearch ./exported.json

# Check export
wc -l exported.json
grep -c '"content"' exported.json  # Count chunks
```

### Step 4: Import to pgvector

```bash
# Set connection string
export PGVECTOR_CONNECTION="postgresql://user:pass@localhost:5432/knowledge"

# Import with same model
sw-search ./exported.json \
  --chunking-strategy json \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs \
  --model mini  # Match original model
```

### Step 5: Verify Migration

```bash
# Test search on pgvector
sw-search search \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs \
  --model mini \
  "test query"
```

Compare results to original SQLite queries.

### Step 6: Update Agent Configuration

**Before (SQLite):**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search documentation",
    "index_path": "./knowledge.swsearch"
})
```

**After (pgvector):**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search documentation",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "docs",
    "model_name": "mini"
})
```

### Step 7: Deploy and Test

```bash
# Test with swaig-test
swaig-test agent.py --exec search_docs --query "test"

# Deploy
python agent.py
```

### Step 8: Create Indexes (Performance)

```sql
-- Connect to database
\c knowledge

-- Create vector index
CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Create metadata index
CREATE INDEX ON knowledge_chunks USING gin (metadata jsonb_path_ops);

-- Create collection index
CREATE INDEX ON knowledge_chunks (collection_name);
```

### Rollback Plan

Keep SQLite file until migration is verified:

```python
# Keep both backends during transition
self.add_skill("native_vector_search", {
    "tool_name": "search_docs_new",
    "description": "Search documentation (new pgvector)",
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "docs",
    "model_name": "mini"
})

self.add_skill("native_vector_search", {
    "tool_name": "search_docs_old",
    "description": "Search documentation (old SQLite)",
    "index_path": "./knowledge.swsearch"
})
```

Test both, compare results, then remove old.

---

## pgvector to SQLite

**When to migrate:**

- Simplifying deployment
- Serverless/Lambda deployment
- Single agent use case
- Reducing infrastructure

### Step 1: Export from pgvector

```bash
# Export collection
sw-search export \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs \
  ./exported.json
```

### Step 2: Build SQLite Index

```bash
# Rebuild as .swsearch file
sw-search ./exported.json \
  --chunking-strategy json \
  --model mini \
  --output knowledge.swsearch
```

### Step 3: Verify

```bash
# Test queries
sw-search search ./knowledge.swsearch "test query"
```

### Step 4: Update Agent

**Before (pgvector):**
```python
{
    "backend": "pgvector",
    "connection_string": os.getenv("PGVECTOR_CONNECTION"),
    "collection_name": "docs",
    "model_name": "mini"
}
```

**After (SQLite):**
```python
{
    "index_path": "./knowledge.swsearch"
}
```

### Step 5: Simplify Dependencies

```dockerfile
# Before: needed PostgreSQL connection
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libpq-dev
RUN pip install signalwire-agents[search-queryonly] psycopg2-binary
ENV PGVECTOR_CONNECTION=postgresql://...
COPY agent.py /app/

# After: standalone
FROM python:3.11-slim
RUN pip install signalwire-agents[search-queryonly]
COPY agent.py knowledge.swsearch /app/
```

Smaller, simpler deployment.

---

## Between pgvector Collections

**When to migrate:**

- Renaming collections
- Consolidating collections
- Creating collection variants
- Testing new configurations

### Option 1: SQL Copy (Fast)

```sql
-- Copy entire collection
INSERT INTO knowledge_chunks (collection_name, content, embedding, metadata)
SELECT 'new_collection', content, embedding, metadata
FROM knowledge_chunks
WHERE collection_name = 'old_collection';

-- Verify
SELECT collection_name, count(*)
FROM knowledge_chunks
GROUP BY collection_name;
```

### Option 2: Export/Import (Flexible)

```bash
# Export old collection
sw-search export \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name old_collection \
  ./exported.json

# Import as new collection
sw-search ./exported.json \
  --chunking-strategy json \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name new_collection \
  --model mini
```

### Rename Collection

```sql
-- Simple rename
UPDATE knowledge_chunks
SET collection_name = 'new_name'
WHERE collection_name = 'old_name';
```

### Delete Old Collection

```sql
-- After verifying new collection works
DELETE FROM knowledge_chunks
WHERE collection_name = 'old_collection';

-- Vacuum to reclaim space
VACUUM FULL;
```

---

## Changing Embedding Models

**When to migrate:**

- Upgrading from mini to base (quality)
- Downgrading from base to mini (performance)
- Testing different models

**Important:** Cannot reuse embeddings. Must rebuild entire index.

### Step 1: Export Content

```bash
# Export existing index (preserves content + metadata)
sw-search export ./knowledge.swsearch ./exported.json
```

### Step 2: Rebuild with New Model

```bash
# Rebuild with base model
sw-search ./exported.json \
  --chunking-strategy json \
  --model base \
  --output knowledge_base.swsearch
```

### Step 3: Compare Quality

```bash
# Test same queries on both
echo "test query" | while read query; do
  echo "Mini model:"
  sw-search search ./knowledge.swsearch "$query" --verbose

  echo "Base model:"
  sw-search search ./knowledge_base.swsearch "$query" --verbose
done
```

### Step 4: Update Agent

```python
{
    "index_path": "./knowledge_base.swsearch",
    # Remove model_name - detected from index
}
```

### Benchmark Before Switching

```python
import time

# Test queries
queries = ["query 1", "query 2", "query 3"]

# Time mini model
start = time.time()
for q in queries:
    mini_results = search_mini(q)
mini_time = time.time() - start

# Time base model
start = time.time()
for q in queries:
    base_results = search_base(q)
base_time = time.time() - start

print(f"Mini: {mini_time:.2f}s")
print(f"Base: {base_time:.2f}s")
print(f"Ratio: {base_time/mini_time:.2f}x")

# Compare quality
for q, mini, base in zip(queries, mini_results, base_results):
    print(f"\nQuery: {q}")
    print(f"Mini top result: {mini[0]['similarity']:.3f}")
    print(f"Base top result: {base[0]['similarity']:.3f}")
```

---

## Changing Chunking Strategies

**When to migrate:**

- Current strategy gives poor results
- Content type changed
- Found better strategy for your use case

### Step 1: Analyze Current Strategy

```bash
# Export current chunks
sw-search export ./knowledge.swsearch ./current_chunks.json

# Examine chunk boundaries
python -c "
import json
with open('current_chunks.json') as f:
    data = json.load(f)
    for i, chunk in enumerate(data['chunks'][:5]):
        print(f'Chunk {i}:')
        print(chunk['content'][:200])
        print('---')
"
```

Are chunks too small? Too large? Splitting code? Mixing topics?

### Step 2: Rebuild with New Strategy

```bash
# Try markdown strategy (good for code docs)
sw-search ./docs \
  --chunking-strategy markdown \
  --model mini \
  --output knowledge_markdown.swsearch
```

### Step 3: Compare Results

```bash
# Same queries, different strategies
QUERY="how to authenticate"

echo "Original (sentence):"
sw-search search ./knowledge.swsearch "$QUERY" --count 3

echo "\nNew (markdown):"
sw-search search ./knowledge_markdown.swsearch "$QUERY" --count 3
```

### Step 4: A/B Test in Production

```python
class ABTestAgent(AgentBase):
    def __init__(self):
        super().__init__(name="ABTest")

        # Old strategy
        self.add_skill("native_vector_search", {
            "tool_name": "search_old",
            "description": "Search docs (old chunking)",
            "index_path": "./knowledge_sentence.swsearch"
        })

        # New strategy
        self.add_skill("native_vector_search", {
            "tool_name": "search_new",
            "description": "Search docs (new chunking)",
            "index_path": "./knowledge_markdown.swsearch"
        })
```

Monitor which gives better results, then switch fully.

### Common Strategy Migrations

**Sentence → Markdown (for code docs):**
```bash
sw-search ./docs \
  --chunking-strategy markdown \
  --model mini \
  --output docs_improved.swsearch
```

**Paragraph → QA (for FAQ):**
```bash
sw-search ./faq \
  --chunking-strategy qa \
  --model mini \
  --output faq_improved.swsearch
```

**Any → JSON (for manual curation):**
```bash
# Export current
sw-search export ./knowledge.swsearch ./chunks.json

# Edit chunks.json manually
# Fix chunk boundaries, add metadata

# Rebuild
sw-search ./chunks.json \
  --chunking-strategy json \
  --model mini \
  --output knowledge_curated.swsearch
```

---

## From Other Vector Databases

### From Pinecone

**Step 1: Export from Pinecone**

```python
import pinecone
import json

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("knowledge")

# Fetch all vectors (paginate for large indexes)
vectors = []
for ids in index.list(namespace=""):  # Get all IDs
    fetch_response = index.fetch(ids=ids)
    vectors.extend(fetch_response['vectors'].values())

# Extract content and metadata
chunks = []
for vec in vectors:
    chunks.append({
        "content": vec['metadata']['content'],
        "metadata": {
            k: v for k, v in vec['metadata'].items()
            if k != 'content'
        }
    })

# Save
with open('pinecone_export.json', 'w') as f:
    json.dump({"chunks": chunks}, f, indent=2)
```

**Step 2: Rebuild Indexes**

```bash
# Build .swsearch file
sw-search ./pinecone_export.json \
  --chunking-strategy json \
  --model mini \
  --output knowledge.swsearch
```

**Note:** You'll regenerate embeddings (Pinecone uses different model).

### From Weaviate

**Step 1: Export from Weaviate**

```python
import weaviate
import json

client = weaviate.Client("http://localhost:8080")

# Query all objects
result = client.query.get(
    "Document",
    ["content", "metadata"]
).with_limit(10000).do()

# Convert to chunks format
chunks = []
for doc in result['data']['Get']['Document']:
    chunks.append({
        "content": doc['content'],
        "metadata": doc.get('metadata', {})
    })

# Save
with open('weaviate_export.json', 'w') as f:
    json.dump({"chunks": chunks}, f, indent=2)
```

**Step 2: Rebuild**

```bash
sw-search ./weaviate_export.json \
  --chunking-strategy json \
  --model mini \
  --output knowledge.swsearch
```

### From OpenAI Assistants

**Step 1: Download Files**

```python
from openai import OpenAI
import os

client = OpenAI()

# List files
files = client.files.list(purpose="assistants")

# Download each
for file in files.data:
    content = client.files.content(file.id)
    with open(f"downloaded/{file.filename}", "wb") as f:
        f.write(content.read())
```

**Step 2: Build Index**

```bash
# Build from downloaded files
sw-search ./downloaded \
  --chunking-strategy markdown \
  --model mini \
  --output knowledge.swsearch
```

**Cost savings:** $6,000/month → $0 (self-hosted)

### From LangChain

If you have LangChain code:

**Before (LangChain + Pinecone):**
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(docs, embeddings, index_name="knowledge")
results = vectorstore.similarity_search("query")
```

**After (SignalWire):**
```python
# Build index once
# sw-search ./docs --output knowledge.swsearch

# In agent
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search documentation",
    "index_path": "./knowledge.swsearch"
})

# Agent automatically uses it
```

Extract your documents from LangChain's loaders, then build with sw-search.

---

## Updating Production Indexes

### Zero-Downtime Updates

**Strategy: Blue-Green Collections**

```bash
# Current: collection "docs_v1" (green)
# Build new: collection "docs_v2" (blue)

# Build updated index
sw-search ./updated-docs \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs_v2 \
  --model mini

# Test new collection
sw-search search \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs_v2 \
  --model mini \
  "test queries"

# Switch agents to new collection (rolling update)
kubectl set env deployment/agent \
  COLLECTION_NAME=docs_v2

# After verification, delete old
# DELETE FROM knowledge_chunks WHERE collection_name = 'docs_v1';
```

### Incremental Updates

**For small changes:**

```bash
# Export current
sw-search export \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs \
  ./current.json

# Edit current.json - add/update/remove chunks

# Rebuild
sw-search ./current.json \
  --chunking-strategy json \
  --backend pgvector \
  --connection-string "$PGVECTOR_CONNECTION" \
  --collection-name docs_updated \
  --model mini

# Test and switch
```

### Staged Rollout

```python
import random

class StagedRolloutAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Staged")

        # 90% use old, 10% use new
        if random.random() < 0.1:
            collection = "docs_v2"  # New
        else:
            collection = "docs_v1"  # Old

        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search documentation",
            "backend": "pgvector",
            "connection_string": os.getenv("PGVECTOR_CONNECTION"),
            "collection_name": collection,
            "model_name": "mini"
        })

        logger.info(f"Using collection: {collection}")
```

Monitor metrics, gradually increase percentage.

---

## Migration Checklist

### Pre-Migration

- [ ] Backup current index/data
- [ ] Document current configuration (model, strategy, params)
- [ ] Test current performance (latency, quality)
- [ ] Create test queries for validation
- [ ] Plan rollback strategy

### During Migration

- [ ] Export existing data
- [ ] Validate export (count chunks, sample content)
- [ ] Build new index/collection
- [ ] Test new index with same queries
- [ ] Compare results (quality, performance)
- [ ] Update agent configuration
- [ ] Test agent with new configuration

### Post-Migration

- [ ] Monitor query latency
- [ ] Track no-result queries
- [ ] Compare quality metrics
- [ ] Verify all features work
- [ ] Update documentation
- [ ] Clean up old indexes/collections
- [ ] Update deployment scripts

### Rollback Procedure

If migration has issues:

1. Keep old index/collection available
2. Revert agent configuration
3. Redeploy agents
4. Investigate issues
5. Fix and retry migration

```python
# Emergency rollback
self.add_skill("native_vector_search", {
    "index_path": "./knowledge_old.swsearch"  # Revert
})
```

---

## Common Migration Issues

### Issue: Different Result Quality

**Symptoms:** New index returns different results

**Causes:**

- Different embedding model used
- Different chunking strategy
- Metadata not preserved

**Fix:**
```bash
# Verify model matches
sw-search validate ./old.swsearch  # Check model
sw-search validate ./new.swsearch  # Check model

# Ensure model matches during rebuild
sw-search ./content --model mini  # Explicit model
```

### Issue: Performance Degradation

**Symptoms:** Queries slower after migration

**Causes:**

- Missing pgvector indexes
- Wrong model (mini → base)
- Network latency (local → remote)

**Fix:**
```sql
-- Add indexes
CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops);
```

### Issue: Missing Content

**Symptoms:** Some chunks not in new index

**Causes:**

- Export incomplete
- File filtering excluded content
- Chunking strategy skipped content

**Fix:**
```bash
# Compare chunk counts
sw-search validate ./old.swsearch  # Note count
sw-search validate ./new.swsearch  # Compare

# If different, investigate export
sw-search export ./old.swsearch ./export.json
grep -c '"content"' export.json  # Count in export
```

---

## Key Takeaways

1. **Always backup** - Keep old index until verified
2. **Export preserves everything** - Content and metadata
3. **Models must match** - Can't reuse embeddings across models
4. **Test before switching** - Use same queries to compare
5. **Blue-green for zero downtime** - Parallel collections during transition
6. **Staged rollout reduces risk** - Gradual percentage switch
7. **Monitor post-migration** - Track latency and quality
8. **Have rollback plan** - Quick revert if issues

Migrations are straightforward with proper planning and testing.
