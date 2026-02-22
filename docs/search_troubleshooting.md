# Search Troubleshooting

This document covers common issues, debugging techniques, and best practices for the SignalWire Agents SDK search system. For an overview of search concepts, see [search_overview.md](search_overview.md). For indexing configuration, see [search_indexing.md](search_indexing.md).

## Common Issues

### No Results Returned

**Symptoms:**

The search returns empty results for queries that should match content in the index.

```python
Query: "how to authenticate"
Results: []
```

**Diagnosis:**

1. Verify the index exists and has content:
   ```bash
   ls -lh knowledge.swsearch
   sw-search validate ./knowledge.swsearch
   ```

2. Test the query directly via CLI:
   ```bash
   sw-search search ./knowledge.swsearch "how to authenticate" --verbose
   ```
   Examine the similarity scores in verbose output.

3. Check the distance threshold:
   ```python
   {
       "distance_threshold": 0.7  # Very strict -- may filter out valid results
   }
   ```

**Common Causes and Solutions:**

1. **Distance threshold too strict** -- Lower the threshold to allow weaker but valid matches:
   ```python
   # Problem
   {"distance_threshold": 0.7}  # Only near-perfect matches

   # Solution
   {"distance_threshold": 0.4}  # More permissive
   ```

2. **Content does not exist in the index** -- Verify and add missing content:
   ```bash
   sw-search search ./knowledge.swsearch "authentication" --verbose --count 10
   # If no content about authentication exists, rebuild with additional sources
   sw-search ./auth-docs --output updated.swsearch
   ```

3. **Query phrasing does not match content semantically** -- Try synonym variations:
   ```bash
   # "how to authenticate" may not match content phrased as "bearer token usage"
   sw-search search ./knowledge.swsearch "bearer token" --verbose
   ```

4. **Model mismatch between index build and query** -- The embedding model used at query time must match the model used when building the index:
   ```python
   # Built with base model
   # sw-search ./docs --model base --output docs.swsearch

   # Query must also use base model
   {"model_name": "base"}  # Must match the build model
   ```

**Fallback strategy** -- Try progressively lower thresholds:
```python
def search_with_fallback(self, query):
    """Try multiple thresholds"""
    for threshold in [0.5, 0.4, 0.3]:
        results = self.search(query, threshold=threshold)
        if results:
            return results
    return []
```

---

### Irrelevant Results

**Symptoms:**

Results are returned but do not match the query intent.

```python
Query: "Python authentication examples"
Results:
1. "Ruby authentication guide" (similarity: 0.65)
2. "General API overview" (similarity: 0.58)
3. "Installation instructions" (similarity: 0.52)
```

**Diagnosis:**

1. Check similarity scores:
   ```bash
   sw-search search ./knowledge.swsearch "Python authentication examples" --verbose
   ```
   Scores below 0.5 indicate weak matches.

2. Examine the returned content:
   ```bash
   sw-search search ./knowledge.swsearch "Python authentication examples" --verbose --count 10
   ```

3. Check chunking quality by exporting the index:
   ```bash
   sw-search export ./knowledge.swsearch ./exported.json
   # Review chunk boundaries -- are they mixing topics?
   ```

**Common Causes and Solutions:**

1. **Threshold too permissive** -- Raise the threshold to reject weak matches:
   ```python
   # Problem
   {"distance_threshold": 0.2}  # Accepts weak matches

   # Solution
   {"distance_threshold": 0.4}
   ```

2. **Poor chunking strategy** -- Use a strategy that matches your content type:
   ```bash
   # Problem: sentence strategy splits code blocks
   sw-search ./docs --chunking-strategy sentence

   # Solution: markdown strategy preserves code blocks
   sw-search ./docs --chunking-strategy markdown
   ```
   See [search_indexing.md](search_indexing.md) for details on chunking strategies.

3. **Missing metadata and tags** -- Add tags during indexing to improve hybrid scoring:
   ```bash
   sw-search ./docs --tags python,authentication,examples --output docs.swsearch
   ```

4. **Content lacks specificity** -- If generic content dominates, add more specific documents covering the exact topics users ask about.

**Tag-based filtering:**
```python
self.add_skill("native_vector_search", {
    "tool_name": "search_python_examples",
    "description": "Search for Python code examples",
    "index_path": "./docs.swsearch",
    "tags": ["python", "code"],
    "distance_threshold": 0.4
})
```

---

### Poor Ranking Quality

**Symptoms:**

The best result appears lower in the list instead of first.

```python
Query: "voice configuration"
Results:
1. "General configuration overview" (similarity: 0.72)
2. "Database configuration" (similarity: 0.68)
3. "Perfect voice config guide" (similarity: 0.85)  # Should be #1
```

**Diagnosis:**

1. Check hybrid scoring via verbose output:
   ```bash
   sw-search search ./knowledge.swsearch "voice configuration" --verbose
   ```
   Determine whether the result with higher vector similarity has a lower hybrid score.

2. Examine metadata on each result:
   ```bash
   sw-search export ./knowledge.swsearch ./exported.json
   # Check if the best result lacks metadata that other results have
   ```

**Common Causes and Solutions:**

1. **Missing metadata on the best result** -- Results with tags receive keyword boosts in hybrid scoring. Add tags to the relevant chunks:
   ```bash
   sw-search export ./knowledge.swsearch ./chunks.json
   # Edit chunks.json to add metadata:
   # {"content": "Voice configuration guide...", "metadata": {"tags": ["voice", "configuration", "guide"]}}
   sw-search ./chunks.json --chunking-strategy json --output fixed.swsearch
   ```

2. **Keyword matches boosting wrong results** -- A result containing both query keywords (e.g., "Configuration for voice, database, and API...") may receive a keyword boost even though it is less relevant than a focused document.

3. **Increase result count** -- Returning more results allows the LLM to select the most relevant one:
   ```python
   {"count": 5}
   ```

4. **Custom reranking via formatter callback:**
   ```python
   def _format_with_reranking(self, response, agent, query, results, **kwargs):
       """Rerank based on custom logic"""
       if "example" in query.lower():
           results.sort(key=lambda r: r.get('metadata', {}).get('has_code', False), reverse=True)
       return self._build_response(results)
   ```

---

### Content Truncation

**Symptoms:**

Search results appear cut off mid-sentence or are incomplete.

**Diagnosis:**

1. Check the content length budget:
   ```python
   {"max_content_length": 16384}  # May be too small for the number of results
   ```

2. Examine individual chunk sizes:
   ```bash
   sw-search export ./knowledge.swsearch ./chunks.json
   # Check for very long chunks
   ```

3. Check result count relative to budget:
   ```python
   {
       "count": 10,
       "max_content_length": 16384  # Only ~1.6KB per result
   }
   ```

**Common Causes and Solutions:**

1. **max_content_length too low** -- Increase the budget:
   ```python
   {"max_content_length": 32768}  # 32KB (default)
   ```

2. **Individual chunks too long** -- Rechunk with a strategy that produces smaller pieces:
   ```bash
   sw-search ./docs \
     --chunking-strategy sentence \
     --max-chunk-size 1000 \
     --output docs_small_chunks.swsearch
   ```

3. **Too many results for the budget** -- Reduce the result count so each result gets more space:
   ```python
   {
       "count": 3,
       "max_content_length": 16384
   }
   ```

---

### Function Not Being Called by AI

**Symptoms:**

The agent does not invoke the search function even when the user asks questions that should trigger a search.

**Diagnosis:**

1. Verify the function is registered:
   ```bash
   swaig-test agent.py --list-tools
   # Should list search_docs or the configured tool_name
   ```

2. Test the function directly:
   ```bash
   swaig-test agent.py --exec search_docs --query "test"
   ```

3. Check the function description -- a vague description makes it less likely the LLM will invoke it.

**Common Causes and Solutions:**

1. **Vague function description** -- Provide a specific, detailed description:
   ```python
   # Problem
   {"description": "Search docs"}

   # Solution
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

2. **No prompt instructions telling the agent to search** -- Add explicit guidance:
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

3. **LLM believes it already knows the answer** -- Strengthen the prompt:
   ```python
   "ALWAYS search before answering, even if you think you know the answer"
   ```

See [search_integration.md](search_integration.md) for full details on agent configuration.

---

### Model Loading Errors

**Symptoms:**

```
Error loading model: sentence-transformers/all-MiniLM-L6-v2
```

**Diagnosis:**

1. Check that search dependencies are installed:
   ```bash
   pip list | grep sentence-transformers
   ```

2. Check the model cache:
   ```bash
   ls ~/.cache/huggingface/transformers/
   ```

3. Check network access to Hugging Face:
   ```bash
   curl -I https://huggingface.co
   ```

**Common Causes and Solutions:**

1. **Search dependencies not installed** -- Install the appropriate search extras:
   ```bash
   # Base install does not include search
   pip install signalwire-agents[search]
   ```

2. **Network issues preventing model download on first run** -- Pre-download the model:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

3. **Insufficient disk space** -- Models require approximately 400MB:
   ```bash
   df -h ~/.cache
   # Clean if needed
   rm -rf ~/.cache/huggingface/transformers/
   ```

**Pre-download in Docker builds:**
```dockerfile
RUN pip install signalwire-agents[search] && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Offline mode** -- If the model is already cached:
```bash
export TRANSFORMERS_OFFLINE=1
```

**Query-only mode** -- If you only need to query existing indexes and do not need the full model:
```bash
pip install signalwire-agents[search-queryonly]
```

---

### "Illegal Instruction" Error (CPU Compatibility)

**Symptoms:**

When running `sw-search` with embedding generation, the process crashes with an "Illegal instruction" error.

**Root Cause:**

This occurs when PyTorch was compiled with newer CPU instruction sets (such as AVX2 or AVX-512) that are not supported by the host CPU. This is common on older server hardware or certain virtualized environments.

**Diagnosis:**

1. Check CPU capabilities:
   ```bash
   cat /proc/cpuinfo | grep -E "(model name|flags)" | head -5
   ```
   Look for `avx2` and `avx512f` in the flags. If absent, the CPU does not support these instruction sets.

2. Check PyTorch version:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

3. Test model loading to reproduce the error:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
   ```

**Solution:**

Set environment variables to disable the unsupported instruction sets:

```bash
export PYTORCH_DISABLE_AVX2=1
export PYTORCH_DISABLE_AVX512=1
```

Then run commands with these variables set:
```bash
PYTORCH_DISABLE_AVX2=1 PYTORCH_DISABLE_AVX512=1 sw-search ./docs --output index.swsearch
```

**Permanent Fix:**

Add the exports to your shell profile (`.bashrc`, `.zshrc`, or equivalent):
```bash
export PYTORCH_DISABLE_AVX2=1
export PYTORCH_DISABLE_AVX512=1
```

---

### pgvector Connection Issues

**Symptoms:**

```
Error: could not connect to PostgreSQL server
```

or

```
vector type not found in the database
```

**Diagnosis:**

1. Test the connection directly:
   ```bash
   psql "$PGVECTOR_CONNECTION"
   ```

2. Check the connection string format:
   ```bash
   # Required format: postgresql://user:pass@host:port/db
   echo "$PGVECTOR_CONNECTION"
   ```

3. Check whether the pgvector extension is installed:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

4. Check whether PostgreSQL is running:
   ```bash
   sudo systemctl status postgresql
   sudo systemctl start postgresql  # Start if needed
   ```

**Common Causes and Solutions:**

1. **Incomplete connection string:**
   ```python
   # Problem: missing components
   "postgresql://localhost/db"

   # Solution: include all components
   "postgresql://user:pass@localhost:5432/db"
   ```

2. **pgvector extension not installed:**
   ```bash
   # Install the extension package (PostgreSQL 16 example)
   sudo apt-get install postgresql-16-pgvector
   ```
   ```sql
   -- Enable in the database (as superuser)
   CREATE EXTENSION vector;

   -- Verify
   \dx
   ```

   Alternatively, use the Docker image with pgvector pre-installed:
   ```bash
   docker run -d \
     --name postgres-pgvector \
     -e POSTGRES_USER=signalwire \
     -e POSTGRES_PASSWORD=signalwire123 \
     -e POSTGRES_DB=knowledge \
     -p 5432:5432 \
     pgvector/pgvector:pg16
   ```

3. **Database does not exist:**
   ```bash
   createdb -U postgres knowledge
   ```

4. **Network or firewall blocking the connection:**
   ```bash
   telnet postgres-host 5432
   ```

5. **Authentication failure** -- Verify credentials and check `pg_hba.conf` for allowed connection methods.

6. **Docker container cannot reach host PostgreSQL:**
   ```bash
   # Linux: use host networking
   docker run --network host ...

   # macOS/Windows: use host.docker.internal
   # postgresql://user:pass@host.docker.internal:5432/knowledge

   # Or use a Docker network
   docker network create myapp
   docker run -d --name postgres --network myapp pgvector/pgvector:pg16
   docker run --network myapp -e DB_HOST=postgres ...
   ```

**Validate connection string programmatically:**
```python
import os
conn = os.getenv("PGVECTOR_CONNECTION")
assert conn.startswith("postgresql://"), "Invalid connection string"
assert "@" in conn, "Missing credentials"
assert ":" in conn.split("@")[1], "Missing port"
```

**Test connection:**
```python
from sqlalchemy import create_engine

engine = create_engine(os.getenv("PGVECTOR_CONNECTION"))
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print("Connected successfully")
```

See [search_deployment.md](search_deployment.md) for full pgvector deployment guidance.

---

### Slow Queries

**Symptoms:**

Search queries take 500ms or more instead of the expected 20-50ms range.

**Diagnosis:**

1. Profile query time:
   ```bash
   time sw-search search ./knowledge.swsearch "test query"
   ```

2. Check index size:
   ```bash
   ls -lh knowledge.swsearch

   # For pgvector
   # SELECT count(*) FROM knowledge_chunks;
   ```

3. Check which model is being used:
   ```python
   {"model_name": "large"}  # Large models are significantly slower
   ```

**Common Causes and Solutions:**

1. **Using the large model** -- Switch to the mini model for 2-3x faster queries with minimal quality loss:
   ```python
   {"model_name": "mini"}
   ```

2. **No vector index on pgvector** -- Create an IVFFlat index:
   ```sql
   -- For small datasets (< 100k rows)
   CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
     WITH (lists = 100);

   -- For larger datasets
   CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
     WITH (lists = 1000);
   ```

3. **Too many results requested** -- Reduce the result count:
   ```python
   {"count": 3}  # Faster than requesting 10+
   ```

4. **Remote pgvector with high network latency** -- For single-agent deployments, consider using a local SQLite index instead:
   ```python
   {"index_path": "./local.swsearch"}
   ```

5. **PostgreSQL not tuned for vector operations:**
   ```sql
   SET work_mem = '256MB';
   -- Or globally:
   ALTER SYSTEM SET work_mem = '256MB';
   ```

6. **Analyze query plans** to identify bottlenecks:
   ```sql
   EXPLAIN (ANALYZE, BUFFERS)
   SELECT * FROM your_collection
   ORDER BY embedding <=> '[...]'::vector
   LIMIT 5;
   ```

**Connection pool exhaustion** (multi-agent deployments) -- Use pgbouncer for connection pooling:
```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
knowledge = host=localhost port=5432 dbname=knowledge

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```
```bash
# Connect through pgbouncer (default port 6432)
# postgresql://user:pass@localhost:6432/knowledge
```

---

## Debugging

### Debug Mode

Enable verbose logging to see detailed search internals:

```python
import os
os.environ['SIGNALWIRE_LOG_LEVEL'] = 'DEBUG'
```

This exposes similarity scores, hybrid scoring details, and timing information in the agent logs.

### Inspecting Chunks

Export the index to inspect what was actually indexed:

```bash
sw-search export ./knowledge.swsearch ./inspect.json
```

Review the exported JSON to verify:
- Chunk boundaries are sensible and do not split related content
- Metadata and tags are present on chunks that need them
- Chunk sizes are appropriate (not too large, not too small)

### Checking Embeddings

Verify that the embedding model loads correctly and produces vectors:

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(m.encode('test').shape)"
```

If this fails, see the [Model Loading Errors](#model-loading-errors) and ["Illegal Instruction" Error](#illegal-instruction-error-cpu-compatibility) sections above.

### Score Analysis

Use verbose CLI output to analyze similarity scores:

```bash
sw-search search ./knowledge.swsearch "your query" --verbose --count 10
```

Interpret the scores:
- **0.7 and above**: Strong semantic match
- **0.5 to 0.7**: Moderate match, usually relevant
- **0.3 to 0.5**: Weak match, may or may not be relevant
- **Below 0.3**: Poor match, likely irrelevant

A typical `distance_threshold` of 0.4 provides a good balance between recall and precision.

### Verbose CLI Output

The `--verbose` flag on `sw-search search` provides:
- Similarity scores for each result
- Hybrid scoring breakdown (vector score, keyword boost, metadata boost)
- Query timing information
- Number of chunks scanned

```bash
sw-search search ./knowledge.swsearch "query" --verbose
```

---

## Debugging Checklist

When troubleshooting search issues, work through this checklist in order:

- Verify the index file exists and passes validation (`sw-search validate`)
- Test the query with the CLI tool (`sw-search search --verbose`)
- Check the distance threshold (0.4 is a typical starting value)
- Verify the model matches between build and query time
- Examine chunk quality by exporting and reviewing the index
- Check that the function description is specific and clear
- Verify prompt instructions tell the agent when to use search
- Monitor search latency
- Check for missing Python dependencies
- Test pgvector connection if using that backend
- Review verbose output for similarity scores
- Verify metadata and tags are present on indexed chunks

---

## Best Practices Summary

### Index Building

- **Choose the right chunking strategy for your content type:**
  - Technical docs with code: `markdown`
  - FAQ/support content: `qa`
  - General articles/blogs: `paragraph`
  - Academic papers: `semantic`
  - Mixed content: start with `markdown`
  - Custom needs: JSON workflow for manual curation
- **Add tags during indexing** to improve hybrid search scoring:
  ```bash
  sw-search ./docs --tags documentation,api,code --output docs.swsearch
  ```
- **Use the mini model** for most use cases. Only upgrade to base or large if quality is insufficient.
- **Validate indexes after building** to catch issues early:
  ```bash
  sw-search validate ./knowledge.swsearch
  ```
- **Test with representative queries** before deploying:
  ```bash
  sw-search search ./knowledge.swsearch "expected user question" --verbose
  ```

### Agent Configuration

- **Write specific, detailed function descriptions** -- The LLM uses the description to decide when to call the search function. Generic descriptions like "Search docs" lead to the function not being invoked.
- **Add explicit prompt instructions** telling the agent to search before answering:
  ```python
  self.prompt_add_section("Instructions", bullets=[
      "Always search before answering technical questions",
      "Base answers on search results, not general knowledge",
      "If no results, say you don't have that information"
  ])
  ```
- **Match the model name** between the index build and the agent configuration.
- **Set a balanced distance threshold** -- Start with 0.4 and adjust based on testing.
- **Tune result count and content length** together to avoid truncation:
  ```python
  {
      "count": 3,
      "max_content_length": 32768
  }
  ```

### Production Deployment

- **Use `search-queryonly` in production** to save approximately 400MB of dependencies:
  ```bash
  pip install signalwire-agents[search-queryonly]
  ```
- **Use `search-full` for building indexes** in your CI/CD pipeline or development environment.
- **Switch to pgvector for multi-agent deployments** -- SQLite is appropriate for single-agent setups; pgvector handles concurrent access and shared knowledge bases.
- **Pre-download models in Docker builds** to avoid first-run delays:
  ```dockerfile
  RUN pip install signalwire-agents[search] && \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
  ```
- **Create pgvector indexes** on the embedding column for large datasets:
  ```sql
  CREATE INDEX ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
  ```
- **Use connection pooling** (pgbouncer) for multi-agent deployments sharing a pgvector database.
- **Monitor search performance** -- Track query latency, result counts, and no-result queries to identify content gaps and performance regressions.

### Common Mistakes to Avoid

1. **Not telling the agent to search** -- Adding the search skill without prompt instructions means the LLM may never invoke it. Always pair `add_skill()` with prompt instructions.

2. **Using the wrong chunking strategy** -- Sentence-based chunking splits code blocks and destroys their meaning. Use the markdown strategy for technical documentation.

3. **Setting the distance threshold too strictly or too loosely:**
   - Too strict (0.7): returns no results for valid queries
   - Too permissive (0.2): returns irrelevant results
   - Recommended starting point: 0.4

4. **Mismatched models** -- Building the index with one model and querying with a different model produces meaningless similarity scores. Always match the model.

5. **Ignoring metadata** -- Indexes built without tags lose the keyword boosting benefits of hybrid search. Add relevant tags during indexing.

6. **Deploying without testing** -- Always test with the CLI and `swaig-test` before deploying:
   ```bash
   sw-search search ./knowledge.swsearch "test queries"
   swaig-test agent.py --exec search_docs --query "test"
   ```

---

## Getting Help

**CLI reference:**
```bash
sw-search --help
sw-search search --help
sw-search export --help
```

**Enable debug logging:**
```python
import os
os.environ['SIGNALWIRE_LOG_LEVEL'] = 'DEBUG'
```

**Test with CLI:**
```bash
sw-search search ./knowledge.swsearch "query" --verbose
```

**Export and inspect index contents:**
```bash
sw-search export ./knowledge.swsearch ./inspect.json
```

**GitHub Issues:**
https://github.com/signalwire/signalwire-agents-sdk/issues

**GitHub Discussions:**
https://github.com/signalwire/signalwire-agents-sdk/discussions
