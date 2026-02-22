# Search System Overview

The SignalWire Agents SDK includes a local search system that provides Retrieval-Augmented Generation (RAG) capabilities without external dependencies. The system uses local embeddings, hybrid search techniques, and portable `.swsearch` index files to enable agents to search through document collections. This document consolidates the foundational concepts, installation, architecture, and usage of the search system.

For related documentation, see:

- [Search Indexing](search_indexing.md) - Building indexes, chunking strategies, embeddings
- [Search Integration](search_integration.md) - Agent integration, skills, and API usage
- [Search Deployment](search_deployment.md) - Production deployment, pgvector, scaling
- [Search Troubleshooting](search_troubleshooting.md) - Common issues and solutions

---

## What Is the Search System

### The Problem: Hallucinations and RAG

Large language models are trained on vast amounts of internet text, but they have no knowledge of your specific documentation, internal knowledge bases, or product details. When asked questions outside their training data, they do not say "I don't know" -- they generate plausible-sounding answers that are often completely fictional.

For AI agents representing a business, this is unacceptable. The agent needs to answer questions accurately based on actual documentation, not fabricate answers.

**Retrieval-Augmented Generation (RAG)** solves this problem:

1. When a user asks a question, search the knowledge base for relevant information.
2. Include that information in the prompt to the LLM.
3. Instruct the LLM to answer based on the retrieved information.

Instead of relying on the model's training data, the LLM becomes a natural language interface to a knowledge base.

Most RAG implementations require setting up a separate vector database (Pinecone, Weaviate, Qdrant, etc.), writing embedding management code, handling retrieval logic, paying for additional infrastructure, and dealing with latency from multiple service calls. The SignalWire Agents SDK search system eliminates these requirements by integrating search directly into the agent framework.

### How It Works: Architecture

```
+-------------------+    +------------------+    +-------------------+
|   Documents       |--->|   Index Builder   |--->|  .swsearch DB     |
| (MD, PDF, etc.)   |    |                  |    |                   |
+-------------------+    +------------------+    +-------------------+
                                                          |
                                                          v
+-------------------+    +------------------+    +-------------------+
|     Agent         |--->|  Search Skill    |--->|  Search Engine    |
|                   |    |                  |    |                   |
+-------------------+    +------------------+    +-------------------+
```

**Indexing phase:** Documents are processed by the `IndexBuilder`, which scans files, extracts text, breaks content into chunks, generates vector embeddings for each chunk, and saves everything into a portable `.swsearch` SQLite database.

**Query phase:** At runtime, an agent equipped with the `native_vector_search` skill sends user queries through the `SearchEngine`, which converts the query to a vector embedding, compares it against all stored chunk vectors, applies hybrid scoring (vector similarity + keyword matching + metadata filtering), and returns ranked results. The LLM then uses these results to generate an accurate response.

### Key Features

- **Offline search** -- No external API calls or internet required at query time.
- **Hybrid search** -- Combines vector similarity and keyword search with metadata filtering.
- **Document processing** -- Supports Markdown, PDF, DOCX, HTML, Excel, PowerPoint, and more.
- **Smart chunking** -- Nine chunking strategies including markdown-aware and semantic chunking.
- **Advanced query processing** -- Optional NLP-enhanced query understanding with synonym expansion.
- **Flexible deployment** -- Local embedded mode with `.swsearch` files, remote server mode, or PostgreSQL pgvector backend.
- **Portable indexes** -- A single `.swsearch` file contains the entire knowledge base (embeddings, metadata, full-text index).
- **Voice-optimized** -- Automatic response formatting adapted for voice conversations vs. text chat.
- **Production-ready backends** -- Start with SQLite for development, scale to pgvector for multi-agent production systems.

---

## Installation

The search system uses optional dependencies to keep the base SDK lightweight. Choose the installation option that matches your use case.

### Installation Options

#### Basic Search (~500MB)

```bash
pip install "signalwire-agents[search]"
```

Includes core search functionality with sentence-transformers for embeddings, scikit-learn, NLTK, numpy, and SQLite FTS5 for keyword search. Supports text and markdown files.

Best for: Local development, CI/CD, resource-constrained environments.

#### Full Document Processing (~600MB)

```bash
pip install "signalwire-agents[search-full]"
```

Adds PDF processing (pdfplumber), DOCX processing (python-docx), Excel/PowerPoint (openpyxl, python-pptx), HTML processing (BeautifulSoup4), and additional file format support (markdown, striprtf, python-magic).

Best for: Production systems that need document processing but prioritize speed.

#### Advanced NLP (~600MB)

```bash
pip install "signalwire-agents[search-nlp]"
```

Adds spaCy for advanced text processing, improved POS tagging, named entity recognition, and enhanced query preprocessing.

**Additional setup required:**

```bash
python -m spacy download en_core_web_sm
```

**Performance note:** Advanced NLP features provide significantly better query understanding and synonym expansion, but are 2-3x slower than basic search. Two NLP backends are available:

- **NLTK (default):** ~50-100ms query processing, good for most use cases.
- **spaCy:** ~150-300ms query processing, better POS tagging and entity recognition, requires model download.

Configure via the `nlp_backend` parameter:

```python
self.add_skill("native_vector_search", {
    "nlp_backend": "nltk"   # Fast, default
})

self.add_skill("native_vector_search", {
    "nlp_backend": "spacy"  # Better quality, slower
})
```

Best for: Applications where search quality is more important than speed.

#### All Search Features (~700MB)

```bash
pip install "signalwire-agents[search-all]"
```

Includes everything above plus pgvector support for PostgreSQL backends.

**Additional setup required:**

```bash
python -m spacy download en_core_web_sm
```

Best for: Full-featured applications with powerful hardware.

### search-queryonly: Production Deployments

For production deployments where agents only need to query existing indexes (not build them):

```bash
pip install "signalwire-agents[search-queryonly]"
```

**Size:** ~400MB -- significantly smaller because it excludes the ML models needed to build indexes.

This option is designed for deploying agents in environments such as Lambda functions, Docker containers, or edge devices where the `.swsearch` index file is pre-built and shipped alongside the agent.

### Feature Comparison Table

| Feature | search | search-full | search-nlp | search-all | pgvector |
|---------|--------|-------------|------------|------------|----------|
| Vector embeddings | Yes | Yes | Yes | Yes | No |
| Keyword search (FTS5) | Yes | Yes | Yes | Yes | No |
| Text files (txt, md) | Yes | Yes | Yes | Yes | No |
| PDF processing | No | Yes | No | Yes | No |
| DOCX processing | No | Yes | No | Yes | No |
| Excel/PowerPoint | No | Yes | No | Yes | No |
| Advanced NLP (spaCy) | No | No | Yes | Yes | No |
| POS tagging | No | No | Yes | Yes | No |
| Named entity recognition | No | No | Yes | Yes | No |
| PostgreSQL support | No | No | No | Yes | Yes |
| Scalable vector search | No | No | No | Yes | Yes |
| Approximate size | ~500MB | ~600MB | ~600MB | ~700MB | varies |

pgvector support can also be installed independently:

```bash
pip install "signalwire-agents[pgvector]"

# Or combined with search:
pip install "signalwire-agents[search,pgvector]"
```

### Verifying Installation

```python
try:
    from signalwire_agents.search import IndexBuilder, SearchEngine
    print("Search functionality is available")
except ImportError as e:
    print(f"Search not available: {e}")
    print("Install with: pip install signalwire-agents[search]")
```

Common installation issues:

| Error | Solution |
|-------|----------|
| `No module named 'sentence_transformers'` | `pip install signalwire-agents[search]` |
| `No module named 'pdfplumber'` | `pip install signalwire-agents[search-full]` |
| NLTK data not found | `import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')` |
| spaCy model not found | `python -m spacy download en_core_web_sm` |

Note: The sentence-transformers library downloads pre-trained models (~400MB) on first use. This is cached locally after the initial download.

---

## Core Concepts

### Vector Embeddings

An embedding is a list of numbers (a vector) that represents the semantic meaning of a piece of text. Embeddings function as coordinates in a high-dimensional meaning space: texts with similar meanings produce vectors that are close together, while unrelated texts produce vectors that are far apart.

Simplified example (real embeddings have hundreds of dimensions):

```
"How do I make a phone call?"     -> [0.2, 0.8, 0.1, ...]
"Initiating voice connections"    -> [0.3, 0.7, 0.2, ...]
"Pizza recipes"                    -> [0.9, 0.1, 0.8, ...]
```

The first two texts are close together in vector space (similar meanings), while the third is far away (different topic).

The embedding models used by the search system (such as `sentence-transformers/all-MiniLM-L6-v2`) are neural networks trained on large text corpora to learn semantic relationships. They understand that:

- "car" and "automobile" are synonyms
- "doctor" is related to "hospital" and "medicine"
- Context matters: "Python" can mean a snake or a programming language

During indexing, the model reads each chunk of text, processes it through the neural network, outputs a vector (typically 384 or 768 numbers), and stores that vector alongside the original text. At query time, the same model converts the user's query into a vector and compares it to all stored chunk vectors.

**Available models:**

| Model | Dimensions | Speed | Quality | Notes |
|-------|-----------|-------|---------|-------|
| MiniLM ("mini", default) | 384 | ~5x faster than base | Good for most use cases | Lower memory usage |
| MPNet ("base") | 768 | Baseline | Better for complex queries | Previous default |

For detailed model comparisons and embedding strategies, see [Search Indexing](search_indexing.md).

### Semantic vs Keyword Search

Traditional keyword search looks for exact word matches. If documentation says "initiating voice connections" and a user asks "how do I make a phone call", keyword search finds nothing -- not a single word matches.

Vector search finds it immediately because the meaning is similar. It understands that:

- "change" is approximately equivalent to "configuring" and "setting"
- "AI's voice" is approximately equivalent to "voice model"
- "speech settings" is approximately equivalent to "speech characteristics"

However, keyword search still matters for exact terms:

- Model names: "gpt-4o-mini" vs "GPT-4"
- API endpoints: "/api/v2/users" vs "/api/users"
- Error codes: "ERROR_404" vs "ERROR_500"
- Programming identifiers: `set_params()` vs `get_params()`

The search system combines both approaches (hybrid search) to handle semantic similarity and exact term matching.

### Cosine Similarity and Distance Threshold

The comparison between query vectors and chunk vectors uses **cosine similarity**, which measures the angle between two vectors. Vectors pointing in the same direction (similar meanings) have high similarity (close to 1.0), while vectors pointing in different directions have low similarity (close to 0.0).

```
Query: "how to handle errors"
Chunk A: "error handling guide"     -> similarity: 0.87 (very similar)
Chunk B: "installation instructions" -> similarity: 0.23 (not similar)
```

The `distance_threshold` parameter filters out low-similarity results. Only chunks with similarity above the threshold are returned:

```python
"distance_threshold": 0.3  # Very strict - only near-perfect matches
"distance_threshold": 0.5  # Balanced - good matches
"distance_threshold": 0.7  # Permissive - includes loosely related content
```

For technical documentation, 0.4-0.5 is a good starting point. For creative content or broad topics, 0.6-0.7 may be more appropriate. Too strict yields no results; too permissive yields irrelevant results.

Once embeddings are generated, search is extremely fast -- comparing vectors is basic arithmetic (multiply and add operations). Modern CPUs can compare thousands of vectors per millisecond. The expensive embedding generation happens once during indexing; queries are cheap.

### Hybrid Search Algorithm

The search system implements a **vector-first** hybrid search algorithm that combines vector similarity, keyword matching, and metadata filtering to produce higher-quality results than any single approach alone.

#### Vector-First Architecture

Many RAG systems use a keyword-first approach: try keyword search first, fall back to vector search if it fails. This treats vector search as a backup and misses semantically relevant content that lacks keyword matches.

The SignalWire search system takes the opposite approach:

1. **Always run vector search** as the primary signal.
2. **Run keyword/metadata searches in parallel** (not conditionally).
3. **Use keyword matches as confirmation signals** that boost vector scores.
4. **Return the best combined results.**

Vector search drives the results. Keyword and metadata matches boost the scores of results that match on multiple signals.

#### The Confirmation Principle

The core insight of hybrid search is **confirmation**: when multiple independent signals agree that a chunk is relevant, confidence in that result increases.

- Vector search is one signal: "This seems semantically related."
- Keyword search is another: "It contains the exact terms."
- Metadata is a third: "It is tagged as relevant."

When all three agree, the result is strongly confirmed.

#### Scoring Mechanics

The algorithm proceeds in these steps:

```
1. Vector search for N*3 candidates (where N = requested count)
   -> Generates pool of semantically similar chunks

2. Parallel keyword search for query terms
   -> Identifies chunks with exact term matches

3. Parallel metadata search for tags/fields
   -> Identifies chunks with matching metadata

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

**Concrete example:**

Query: "how to configure voice settings"

After vector search produces a candidate pool, hybrid scoring adjusts the rankings:

```
Result A: Vector similarity 0.78
  + Keyword matches: "voice", "configuration" -> +15% each (capped at 30%)
  + Metadata tag: "voice" -> +15%
  Final score: 0.78 * 1.30 = 1.01

Result B: Vector similarity 0.82
  + No keyword or metadata matches
  Final score: 0.82

Result A now ranks higher than Result B.
```

**Code chunk boosting:** Chunks containing code blocks receive an additional 20% boost when keyword or metadata matches are also present. This ensures that technical queries surface actual code examples rather than only conceptual descriptions:

```
Result with code: 0.70 (vector) * 1.15 (keyword) * 1.20 (code tag) = 0.97
Result without code: 0.85 (vector) * 1.0 (no boost) = 0.85
-> Code example ranks higher
```

The boost percentages are tuned based on testing with technical documentation. They can be influenced indirectly through `distance_threshold` (stricter thresholds mean fewer candidates for keyword boosting) and `count` (internally the system searches for 3x the requested count to build a good candidate pool before hybrid scoring).

---

## Quick Start

### Building Your First Index

Use the `sw-search` CLI tool to create a `.swsearch` index file from your documents:

```bash
# Build from a directory of documentation
sw-search ./docs --output knowledge.swsearch

# Build from specific files
sw-search README.md docs/agent_guide.md docs/architecture.md --output knowledge.swsearch

# Build from mixed sources (files and directories) with file type filtering
sw-search docs/ examples/ README.md --file-types md,py,txt --output comprehensive.swsearch

# Include verbose output to see progress
sw-search ./docs --output knowledge.swsearch --verbose
```

The `sw-search` command will:

1. Recursively scan the specified sources.
2. Extract text from supported file types (markdown, txt, etc.).
3. Break the content into chunks.
4. Generate vector embeddings for each chunk.
5. Save everything to the output `.swsearch` file.

Example output:

```
Processing files...
  docs/getting-started.md (3 chunks)
  docs/api-reference.md (12 chunks)
  docs/examples.md (5 chunks)

Building index...
  Generated embeddings for 20 chunks
  Index saved to knowledge.swsearch

Index size: 2.3 MB
Total chunks: 20
```

For full CLI reference, see [CLI Guide](cli_guide.md). For advanced indexing options (chunking strategies, model selection, pgvector backend), see [Search Indexing](search_indexing.md).

### Querying from an Agent

Add the `native_vector_search` skill to any agent to enable search:

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

        # Add search capability pointing to a local .swsearch file
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the documentation to answer user questions about features, APIs, and how-to guides",
            "index_file": "./knowledge.swsearch",
            "count": 5,
            "distance_threshold": 0.4,
            "no_results_message": "I couldn't find information about '{query}' in the documentation. Could you rephrase your question?",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search the documentation for that...",
                        "Searching for information..."
                    ]
                }
            }
        })

        # Build the prompt
        self.prompt_add_section(
            "Your Role",
            bullets=[
                "You are a helpful documentation assistant.",
                "Always use search_docs to find information before answering technical questions.",
                "Base your answers on the search results, not general knowledge.",
                "If the search returns no results, admit you don't have that information."
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

When a user asks a question, the following occurs:

1. The query is converted to a vector embedding.
2. The search engine compares the query vector to all chunk vectors in the index.
3. Chunks are scored by cosine similarity.
4. Keyword and metadata matches boost scores (hybrid scoring).
5. Top results (above the distance threshold) are returned.
6. Results are formatted and provided to the LLM.
7. The LLM generates an answer grounded in the retrieved content.

### Testing Your Search

Use the `swaig-test` CLI to verify search functionality without running the full agent:

```bash
# List available tools
swaig-test docs_agent.py --list-tools
# Output: search_docs

# Execute a test search
swaig-test docs_agent.py --exec search_docs --query "how do I create an agent"
```

You can also test the index directly from the command line:

```bash
# Search within an index file
sw-search search knowledge.swsearch "how to create an agent"

# Search with options
sw-search search knowledge.swsearch "API reference" --count 3 --verbose

# JSON output for scripting
sw-search search knowledge.swsearch "error handling" --json | jq '.results[0].content'

# Validate index integrity and view statistics
sw-search validate knowledge.swsearch --verbose
```

---

## Comparison to Alternatives

### vs OpenAI Assistants

OpenAI Assistants provide a hosted file-search solution where documents are uploaded to OpenAI and everything is managed:

```python
from openai import OpenAI
client = OpenAI()

file = client.files.create(file=open("knowledge.pdf", "rb"), purpose="assistants")
assistant = client.beta.assistants.create(
    name="My Assistant",
    instructions="You are a helpful assistant",
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
)
```

**Advantages of OpenAI Assistants:** Zero setup, managed infrastructure, automatic improvements.

**Disadvantages:** Vendor lock-in (OpenAI models only), no control over chunking or search tuning, data privacy concerns (documents reside on OpenAI servers), higher latency (multiple API round-trips), and significantly higher cost for search infrastructure.

**Cost comparison (10,000 queries/month):**

- OpenAI Assistants: ~$1,400-1,500/month (vector storage + inference)
- SignalWire SDK with pgvector: ~$100/month for search infrastructure
- SignalWire SDK with .swsearch: $0/month for search infrastructure

Note: LLM inference costs apply to all approaches and are not included above.

### vs LangChain + External Vector DB

LangChain combined with a managed vector database (Pinecone, Weaviate, Qdrant) provides a flexible but complex approach:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

pinecone.init(api_key="...", environment="...")
loader = DirectoryLoader('./docs')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="knowledge")
results = vectorstore.similarity_search(query, k=5)
```

**Equivalent SignalWire approach:**

```python
agent.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "./knowledge.swsearch"
})
# Agent automatically handles retrieval, formatting, and prompt integration
```

**Advantages of LangChain + VectorDB:** Highly flexible, large community, many integrations.

**Disadvantages:** Complex setup with many moving parts, requires managed vector DB subscription ($70+/month for Pinecone), additional latency from network calls to external services, and manual integration code between retrieval and agent.

### vs DIY Approach

Building a custom solution with an embedding API (OpenAI, Cohere) and custom search logic:

```python
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Generate embeddings (per-call API cost)
doc_embeddings = []
for doc in documents:
    response = openai.Embedding.create(input=doc, model="text-embedding-3-small")
    doc_embeddings.append(response['data'][0]['embedding'])

# Manual storage, retrieval, ranking...
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
top_k = np.argsort(similarities)[-5:][::-1]
```

**Advantages of DIY:** Full control over every aspect, minimal dependencies, good for learning.

**Disadvantages:** Significant development time (estimated 36-72 hours vs 20 minutes for the SDK approach), ongoing maintenance burden, per-call API costs for embeddings, and missing optimizations like hybrid search and metadata boosting.

**Development time comparison:**

| Task | DIY | SignalWire SDK |
|------|-----|----------------|
| Document loading | 2-4 hours | Included |
| Chunking strategies | 4-8 hours | Included (9 strategies) |
| Embedding generation | 2-4 hours | Included |
| Storage layer | 4-8 hours | Included |
| Search implementation | 4-8 hours | Included |
| Hybrid search | 8-16 hours | Included |
| Metadata filtering | 4-8 hours | Included |
| Testing and optimization | 8-16 hours | Included |
| **Total** | **36-72 hours** | **~20 minutes** |

### Feature Matrix

| Feature | OpenAI Assistants | LangChain + VectorDB | DIY | SignalWire SDK |
|---------|-------------------|----------------------|-----|----------------|
| **Setup Time** | 30 min | 2-4 hours | 8-16 hours | 20 min |
| **Chunking Control** | No | Yes | Yes | Yes (9 strategies) |
| **Hybrid Search** | Unknown | Manual | Manual | Built-in |
| **Metadata Filtering** | Limited | Yes | Manual | Yes |
| **Cost (10K queries)** | ~$1,400/mo | $100-200/mo | $50-150/mo | $0-100/mo |
| **Vendor Lock-in** | High | Moderate | None | None |
| **Offline Operation** | No | No | Possible | Yes (.swsearch) |
| **Voice Optimized** | No | No | No | Yes |
| **Agent Integration** | Manual | Manual | Manual | Automatic |
| **Deployment Size** | N/A | Large | Medium | Small (query-only) |
| **Latency** | 100-200ms | 50-130ms | 50-100ms | 5-30ms |
| **LLM Choice** | OpenAI only | Any | Any | Any |
| **Data Privacy** | OpenAI servers | 3rd party | Your infra | Your infra |
| **Scalability** | Auto | Managed | DIY | pgvector |

Note: Cost figures represent search infrastructure only. All approaches have additional LLM inference costs.
