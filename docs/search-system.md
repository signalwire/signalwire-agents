# Local Search System

The SignalWire Agents SDK includes a powerful local search system that provides DataSphere-compatible search functionality without external dependencies. This system uses advanced query preprocessing, local embeddings, and hybrid search techniques to enable agents to search through document collections offline.

## Table of Contents

- [Overview](#overview)
- [Installation Options](#installation-options)
- [Quick Start](#quick-start)
- [Building Search Indexes](#building-search-indexes)
- [Using the Search Skill](#using-the-search-skill)
- [Local vs Remote Modes](#local-vs-remote-modes)
- [Advanced Configuration](#advanced-configuration)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The local search system provides:

- **Offline Search**: No external API calls or internet required
- **Hybrid Search**: Combines vector similarity and keyword search
- **Document Processing**: Supports multiple file formats (Markdown, PDF, DOCX, etc.)
- **Smart Chunking**: Intelligent document segmentation with context preservation
- **Advanced Query Processing**: NLP-enhanced query understanding
- **Flexible Deployment**: Local embedded mode or remote server mode
- **SQLite Storage**: Portable `.swsearch` index files

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Index Builder  │───▶│  .swsearch DB   │
│ (MD, PDF, etc.) │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Agent       │───▶│  Search Skill    │───▶│  Search Engine  │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation Options

The search system uses optional dependencies to keep the base SDK lightweight. Choose the installation option that fits your needs:

### Basic Search (~500MB)
```bash
pip install "signalwire-agents[search]"
```
**Includes:**
- Core search functionality
- Sentence transformers for embeddings
- SQLite FTS5 for keyword search
- Basic document processing (text, markdown)

### Full Document Processing (~600MB)
```bash
pip install "signalwire-agents[search-full]"
```
**Adds:**
- PDF processing (PyPDF2)
- DOCX processing (python-docx)
- HTML processing (BeautifulSoup4)
- Additional file format support

### Advanced NLP Features (~700MB)
```bash
pip install "signalwire-agents[search-nlp]"
```
**Adds:**
- spaCy for advanced text processing
- NLTK for linguistic analysis
- Enhanced query preprocessing
- Language detection

**⚠️ Additional Setup Required:**
```bash
python -m spacy download en_core_web_sm
```

**Performance Note:** Advanced NLP features provide significantly better query understanding, synonym expansion, and search relevance, but are 2-3x slower than basic search. Only recommended if you have sufficient CPU power and can tolerate longer response times.

**NLP Backend Control:** You can choose which NLP backend to use:
- **NLTK (default)**: Fast processing, good for most use cases
- **spaCy**: Better quality but slower, requires model download

Configure via the `nlp_backend` parameter in your search skill.

### All Search Features (~700MB)
```bash
pip install "signalwire-agents[search-all]"
```
**Includes everything above**

**⚠️ Additional Setup Required:**
```bash
python -m spacy download en_core_web_sm
```

**Performance Note:** This includes advanced NLP features which improve search quality but increase response times.

### Minimal Installation (Base SDK only)
```bash
pip install signalwire-agents
```
Search functionality will show helpful error messages when dependencies are missing.

## Quick Start

### 1. Install Dependencies
```bash
pip install "signalwire-agents[search-full]"
```

### 2. Build a Search Index
```bash
# Build from the comprehensive concepts guide
sw-search docs/signalwire_agents_concepts_guide.md --output concepts.swsearch

# Build from multiple individual files
sw-search README.md docs/agent_guide.md docs/architecture.md --output knowledge.swsearch

# Build from mixed sources (files and directories)
sw-search docs/signalwire_agents_concepts_guide.md examples --file-types md,py --output comprehensive.swsearch

# Build from a directory (traditional approach)
sw-search docs --output docs.swsearch

# Include specific file types
sw-search docs --file-types md,txt,py

# Exclude patterns
sw-search docs --exclude "**/test/**,**/__pycache__/**"
```

### 3. Use in Your Agent
```python
from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__()
        
        # Add search capability using the concepts guide
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the comprehensive SDK concepts guide for information",
            "index_file": "concepts.swsearch",
            "count": 5
        })

agent = MyAgent()
agent.serve()
```

### 4. Test the Search
Ask your agent: *"How do I create a new agent?"* and it will search the comprehensive concepts guide to provide detailed answers.

## Building Search Indexes

Search indexes are SQLite databases with the `.swsearch` extension that contain processed documents, embeddings, and search metadata.

### Basic Index Building

```bash
# Build index from the comprehensive concepts guide
sw-search docs/signalwire_agents_concepts_guide.md --output concepts.swsearch

# Build from multiple individual files
sw-search README.md docs/agent_guide.md docs/architecture.md --output knowledge.swsearch

# Build from mixed sources (files and directories)
sw-search docs/signalwire_agents_concepts_guide.md examples --file-types md,py --output comprehensive.swsearch

# Build from a directory (traditional approach)
sw-search docs --output docs.swsearch

# Include specific file types
sw-search docs --file-types md,txt,py

# Exclude patterns
sw-search docs --exclude "**/test/**,**/__pycache__/**"

# JSON-based chunking (for pre-chunked content)
sw-search api_chunks.json --chunking-strategy json --file-types json
```

### Advanced Index Building

```bash
# Full configuration example with multiple sources
sw-search docs/signalwire_agents_concepts_guide.md ./examples README.md \
    --output ./knowledge.swsearch \
    --chunking-strategy sentence \
    --max-sentences-per-chunk 8 \
    --file-types md,txt,rst,py \
    --exclude "**/test/**,**/__pycache__/**" \
    --model sentence-transformers/all-mpnet-base-v2 \
    --tags documentation,api \
    --verbose
```

### Supported File Types

| Format | Extension | Requirements |
|--------|-----------|--------------|
| Markdown | `.md` | Built-in |
| Text | `.txt` | Built-in |
| Python | `.py` | Built-in |
| reStructuredText | `.rst` | Built-in |
| PDF | `.pdf` | `search-full` |
| Word Documents | `.docx` | `search-full` |
| HTML | `.html` | `search-full` |
| JSON | `.json` | Built-in |

### Chunking Strategies

The search system supports multiple chunking strategies to optimize for different use cases:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `sentence` | Groups sentences together (default: 5 per chunk) | General documentation |
| `sliding` | Sliding window with overlap | Long-form content |
| `paragraph` | Splits on paragraph boundaries | Structured documents |
| `page` | Preserves page boundaries | PDFs, presentations |
| `semantic` | Groups semantically similar content | Technical docs |
| `topic` | Groups by topic changes | Mixed content |
| `qa` | Optimized for question-answering | FAQs, tutorials |
| `json` | Uses pre-chunked JSON input | Custom preprocessing |

#### JSON Chunking Strategy

The `json` strategy allows you to provide pre-chunked content in a structured format. This is useful when you need custom control over how documents are split and indexed.

Expected JSON format:
```json
{
  "chunks": [
    {
      "chunk_id": "unique_id",
      "type": "content",
      "content": "The actual text content",
      "metadata": {
        "section": "Introduction",
        "url": "https://example.com/docs/intro",
        "custom_field": "any_value"
      },
      "tags": ["intro", "getting-started"]
    }
  ]
}
```

Example usage:
```bash
# First preprocess your documents into JSON chunks
python your_preprocessor.py input.txt -o chunks.json

# Then build the index using JSON strategy
sw-search chunks.json --chunking-strategy json --file-types json
```

This strategy is particularly useful for:
- API documentation with complex structure
- Documents that need custom parsing logic
- Preserving specific metadata relationships
- Integration with external preprocessing tools

### Index Structure

Each `.swsearch` file contains:

- **Document chunks** with embeddings and metadata
- **Full-text search index** (SQLite FTS5)
- **Configuration** and model information
- **Synonym cache** for query expansion

## Using the Search Skill

The `native_vector_search` skill provides search functionality to your agents.

### Basic Configuration

```python
self.add_skill("native_vector_search", {
    "tool_name": "search_knowledge",
    "description": "Search the knowledge base",
    "index_file": "knowledge.swsearch"
})
```

### Advanced Configuration

### NLP Backend Selection

Choose between NLTK (fast) and spaCy (better quality) for query processing:

```python
# Fast NLTK processing (default)
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "docs.swsearch",
    "nlp_backend": "nltk"  # Fast, good for most use cases
})

# Better quality spaCy processing
self.add_skill("native_vector_search", {
    "tool_name": "search_docs", 
    "index_file": "docs.swsearch",
    "nlp_backend": "spacy"  # Slower but better quality, requires model download
})
```

**Performance Comparison:**
- **NLTK**: ~50-100ms query processing, good synonym expansion
- **spaCy**: ~150-300ms query processing, better POS tagging and entity recognition

### Custom Embedding Models

```python
# Use a different embedding model
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "docs.swsearch",
    "model": "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, faster model
})
```

### Query Enhancement

The system automatically enhances queries using:
- **Language detection**
- **POS tagging** (with NLP dependencies)
- **Synonym expansion** using WordNet
- **Keyword extraction**
- **Vector embeddings**

### Response Customization

```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "docs.swsearch",
    
    # Customize responses for voice calls
    "response_prefix": "Based on the documentation, here's what I found:",
    "response_postfix": "Would you like me to search for more specific information?",
    
    # Custom no-results message
    "no_results_message": "I couldn't find information about '{query}'. Try rephrasing your question.",
    
    # SWAIG function fillers for natural conversation
    "swaig_fields": {
        "fillers": {
            "en-US": [
                "Let me search the documentation",
                "Checking our knowledge base",
                "Looking that up for you"
            ]
        }
    }
})
```

### Tag-Based Filtering

```python
# Only search documents tagged with specific categories
self.add_skill("native_vector_search", {
    "tool_name": "search_api_docs",
    "index_file": "docs.swsearch", 
    "tags": ["api", "reference"],  # Only search API docs
    "description": "Search API reference documentation"
})
```

### Complete Configuration Example

```python
self.add_skill("native_vector_search", {
    # Tool configuration
    "tool_name": "search_docs",
    "description": "Search SDK documentation for detailed information",
    
    # Index configuration
    "index_file": "docs.swsearch",
    "build_index": True,  # Auto-build if missing
    "source_dir": "./docs",  # Source for auto-build
    "file_types": ["md", "txt"],
    
    # Search parameters
    "count": 5,  # Number of results
    "distance_threshold": 0.1,  # Similarity threshold
    "tags": ["documentation"],  # Filter by tags
    
    # NLP backend selection
    "nlp_backend": "nltk",  # or "spacy" for better quality
    
    # Response formatting
    "response_prefix": "Based on the documentation:",
    "response_postfix": "Would you like more details?",
    "no_results_message": "No information found for '{query}'",
    
    # SWAIG configuration
    "swaig_fields": {
        "fillers": {
            "en-US": ["Let me search for that", "Checking the docs"]
        }
    }
})
```

### Multiple Search Instances

You can add multiple search instances for different document collections:

```python
# Documentation search with spaCy for better quality
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "docs.swsearch",
    "nlp_backend": "spacy",
    "description": "Search SDK documentation"
})

# Code examples search with NLTK for speed
self.add_skill("native_vector_search", {
    "tool_name": "search_examples", 
    "index_file": "examples.swsearch",
    "nlp_backend": "nltk",
    "description": "Search code examples"
})
```

## Backend Selection

The search system supports multiple storage backends:

### SQLite Backend (Default)
- **File-based**: Stores indexes in `.swsearch` files
- **Portable**: Single file contains everything
- **Simple**: No external dependencies
- **Best for**: Single-agent deployments, development, small to medium datasets

### PostgreSQL pgvector Backend
- **Server-based**: Uses PostgreSQL with pgvector extension
- **Scalable**: Efficient similarity search with IVFFlat/HNSW indexes
- **Concurrent**: Multiple agents can share the same knowledge base
- **Real-time updates**: Add/remove documents without rebuilding
- **Best for**: Production deployments, multi-agent systems, large datasets

### Choosing a Backend

| Feature | SQLite | pgvector |
|---------|--------|----------|
| Setup complexity | None | Requires PostgreSQL |
| Scalability | Limited | Excellent |
| Concurrent access | Poor | Excellent |
| Update capability | Rebuild required | Real-time |
| Performance (small) | Excellent | Good |
| Performance (large) | Poor | Excellent |
| Deployment | File copy | Database connection |
| Multi-agent | Separate copies | Shared knowledge |

## Local vs Remote Modes

The search skill supports both local and remote operation modes.

### Local Mode (Default)

**Pros:**
- Faster (no network latency)
- Works offline
- Simple deployment
- Lower operational complexity

**Cons:**
- Higher memory usage per agent
- Index files must be distributed with each agent
- Updates require redeploying agents

**Configuration:**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "docs.swsearch",  # Local file
    "nlp_backend": "nltk"  # Choose NLP backend
})
```

### Remote Mode

**Pros:**
- Lower memory usage per agent
- Centralized index management
- Easy updates without redeploying agents
- Better scalability for multiple agents
- Shared resources

**Cons:**
- Network dependency
- Additional infrastructure complexity
- Potential latency

**Configuration:**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "remote_url": "http://localhost:8001",  # Search server
    "index_name": "docs",  # Index name on server
    "nlp_backend": "nltk"  # NLP backend for query preprocessing
})
```

### Running a Remote Search Server

1. **Start the search server:**
```bash
python examples/search_server_standalone.py
```

2. **The server provides HTTP API:**
- `POST /search` - Search the indexes
- `GET /health` - Health check and available indexes  
- `POST /reload_index` - Add or reload an index

3. **Test the API:**
```bash
curl -X POST "http://localhost:8001/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how to create an agent", "index_name": "docs", "count": 3}'
```

### Automatic Mode Detection

The skill automatically detects which mode to use:
- If `remote_url` is provided → Remote mode
- If `index_file` is provided → Local mode
- Remote mode takes priority if both are specified

## Advanced Configuration

### Custom Embedding Models

## CLI Reference

### sw-search Command

```bash
sw-search <source_dir> [options]
```

**Arguments:**
- `source_dir` - Directory containing documents to index

**Options:**
- `--output FILE` - Output .swsearch file (default: `<source_dir>.swsearch`) or collection name for pgvector
- `--backend {sqlite,pgvector}` - Storage backend (default: sqlite)
- `--connection-string STRING` - PostgreSQL connection string for pgvector backend
- `--overwrite` - Overwrite existing collection (pgvector only)
- `--chunk-size SIZE` - Chunk size in characters (default: 500)
- `--chunk-overlap SIZE` - Overlap between chunks (default: 50)
- `--file-types TYPES` - Comma-separated file extensions (default: md,txt,rst)
- `--exclude PATTERNS` - Comma-separated glob patterns to exclude
- `--languages LANGS` - Comma-separated language codes (default: en)
- `--model MODEL` - Embedding model name (default: sentence-transformers/all-mpnet-base-v2)
- `--tags TAGS` - Comma-separated tags to add to all chunks
- `--verbose` - Show detailed progress information
- `--validate` - Validate the created index after building

**Subcommands:**

#### validate - Validate Search Index

```bash
sw-search validate <index_file> [--verbose]
```

Validates an existing .swsearch index file and shows statistics.

#### search - Search Within Index

```bash
sw-search search <index_source> <query> [options]
```

Search within an existing .swsearch index file or pgvector collection. This is useful for:
- Testing search quality and relevance
- Exploring index contents
- Debugging search results
- Scripting and automation

**Search Options:**
- `--backend {sqlite,pgvector}` - Storage backend (default: sqlite)
- `--connection-string STRING` - PostgreSQL connection string for pgvector backend
- `--count COUNT` - Number of results to return (default: 5)
- `--distance-threshold FLOAT` - Minimum similarity score (default: 0.0)
- `--tags TAGS` - Comma-separated tags to filter by
- `--nlp-backend {nltk,spacy}` - NLP backend to use (default: nltk)
- `--verbose` - Show detailed information including index stats
- `--json` - Output results as JSON for scripting
- `--no-content` - Hide content in results (show only metadata)

**Examples:**

```bash
# SQLite Backend (Default)
# Build from the comprehensive concepts guide
sw-search docs/signalwire_agents_concepts_guide.md --output concepts.swsearch

# Build from multiple sources (files and directories)
sw-search docs/signalwire_agents_concepts_guide.md examples README.md \
    --output comprehensive.swsearch \
    --file-types md,py,txt \
    --verbose

# PostgreSQL pgvector Backend
# Build index in PostgreSQL
sw-search ./docs \
    --backend pgvector \
    --connection-string "postgresql://user:pass@localhost/dbname" \
    --output docs_collection \
    --file-types md,txt \
    --verbose

# Overwrite existing pgvector collection
sw-search ./docs \
    --backend pgvector \
    --connection-string "postgresql://user:pass@localhost/dbname" \
    --output docs_collection \
    --overwrite

# Validate an existing index
sw-search validate concepts.swsearch --verbose

# Search within SQLite index
sw-search search concepts.swsearch "how to create an agent"
sw-search search concepts.swsearch "API reference" --count 3 --verbose
sw-search search concepts.swsearch "configuration" --tags documentation --json

# Search within pgvector collection
sw-search search docs_collection "how to create an agent" \
    --backend pgvector \
    --connection-string "postgresql://user:pass@localhost/dbname"

sw-search search docs_collection "configuration" \
    --backend pgvector \
    --connection-string "postgresql://user:pass@localhost/dbname" \
    --count 10 \
    --json

# Use different NLP backends
sw-search search concepts.swsearch "deployment options" --nlp-backend nltk  # Fast
sw-search search concepts.swsearch "deployment options" --nlp-backend spacy  # Better quality

# Advanced search with filtering
sw-search search concepts.swsearch "deployment options" \
    --count 10 \
    --distance-threshold 0.1 \
    --tags "deployment,production" \
    --nlp-backend spacy \
    --verbose

# JSON output for scripting
sw-search search concepts.swsearch "error handling" --json | jq '.results[0].content'

# Build multiple specialized indexes
sw-search docs/signalwire_agents_concepts_guide.md --output concepts.swsearch
sw-search examples --output examples.swsearch --file-types py,md
```

### Index Validation

```bash
# Validate an existing index
python -c "
from signalwire_agents.search import SearchEngine
engine = SearchEngine('docs.swsearch')
print(f'Index stats: {engine.get_stats()}')
"
```

## API Reference

### SearchEngine Class

```python
from signalwire_agents.search import SearchEngine

# Load an index
engine = SearchEngine("docs.swsearch")

# Perform search
results = engine.search(
    query_vector=[...],  # Optional: pre-computed query vector
    enhanced_text="search query",  # Enhanced query text
    count=5,  # Number of results
    distance_threshold=0.0,  # Minimum similarity score
    tags=["documentation"]  # Filter by tags
)

# Get index statistics
stats = engine.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Total files: {stats['total_files']}")
```

### IndexBuilder Class

```python
from signalwire_agents.search import IndexBuilder

# Create index builder
builder = IndexBuilder(
    model_name="sentence-transformers/all-mpnet-base-v2",
    chunk_size=500,
    chunk_overlap=50,
    verbose=True
)

# Build index
builder.build_index(
    source_dir="./docs",
    output_file="docs.swsearch",
    file_types=["md", "txt"],
    exclude_patterns=["**/test/**"],
    tags=["documentation"]
)
```