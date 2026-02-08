# Bug Fixes from Code Audit

This document describes all bugs found and fixed during a comprehensive audit of the `signalwire-agents` codebase. Fixes are grouped by file and tagged with severity.

**Total: ~60 bugs fixed across 30 source files and 1 test file.**

**Test results: Zero regressions introduced. 3 previously-failing tests now pass.**

---

## Table of Contents

- [Critical / High Severity](#critical--high-severity)
  - [swml_service.py — Auth bypass: HTTPException returned instead of raised](#swml_servicepy--auth-bypass-httpexception-returned-instead-of-raised)
  - [web_service.py — Path traversal in static file serving](#web_servicepy--path-traversal-in-static-file-serving)
  - [agent_server.py — Path traversal in static file serving](#agent_serverpy--path-traversal-in-static-file-serving)
  - [web_service.py — XSS in directory listing](#web_servicepy--xss-in-directory-listing)
  - [search_engine.py — SQL injection in metadata search](#search_enginepy--sql-injection-in-metadata-search)
  - [pgvector_backend.py — SQL injection via collection name](#pgvector_backendpy--sql-injection-via-collection-name)
  - [auth_handler.py — Timing attack on credential comparison](#auth_handlerpy--timing-attack-on-credential-comparison)
  - [gateway_service.py — Timing attack on token/credential comparison](#gateway_servicepy--timing-attack-on-token-comparison)
  - [auth_mixin.py — Timing attack in GCF/Azure auth](#auth_mixinpy--timing-attack-in-gcfazure-auth)
  - [web_mixin.py — Security token bypass for all SWAIG functions](#web_mixinpy--security-token-bypass-for-all-swaig-functions)
  - [security_config.py — Password regenerated on every call](#security_configpy--password-regenerated-on-every-call)
  - [search_service.py — Credential exposure in health endpoint](#search_servicepy--credential-exposure-in-health-endpoint)
- [Medium Severity](#medium-severity)
  - [swml_service.py — Auth parsing fails on passwords containing colons](#swml_servicepy--auth-parsing-fails-on-passwords-containing-colons)
  - [swml_service.py — Port 0 and empty host rejected as falsy](#swml_servicepy--port-0-and-empty-host-rejected-as-falsy)
  - [swml_service.py — URL parameter values not encoded](#swml_servicepy--url-parameter-values-not-encoded)
  - [swml_service.py — Document mutation across requests](#swml_servicepy--document-mutation-across-requests)
  - [agent_base.py — Operator precedence bug in auto_answer check](#agent_basepy--operator-precedence-bug-in-auto_answer-check)
  - [agent_base.py — URL parameter values not encoded](#agent_basepy--url-parameter-values-not-encoded)
  - [function_result.py — Falsy response treated as empty](#function_resultpy--falsy-response-treated-as-empty)
  - [function_result.py — TypeError when executing SWML with string content](#function_resultpy--typeerror-when-executing-swml-with-string-content)
  - [function_result.py — Caller dict mutation in execute_swml](#function_resultpy--caller-dict-mutation-in-execute_swml)
  - [data_map.py — "required" key collision with parameter definitions](#data_mappy--required-key-collision-with-parameter-definitions)
  - [data_map.py — Incorrect type hint on foreach()](#data_mappy--incorrect-type-hint-on-foreach)
  - [search_engine.py — FTS5 query escaping uses invalid backslash syntax](#search_enginepy--fts5-query-escaping-uses-invalid-backslash-syntax)
  - [search_engine.py — Boost multiplier not applied to final_score](#search_enginepy--boost-multiplier-not-applied-to-final_score)
  - [search_engine.py — fetchall() called without prior execute()](#search_enginepy--fetchall-called-without-prior-execute)
  - [search_engine.py — SQLite connection leaks on exceptions](#search_enginepy--sqlite-connection-leaks-on-exceptions)
  - [search_service.py — Wrong parameter name passed to SearchRequest](#search_servicepy--wrong-parameter-name-passed-to-searchrequest)
  - [search_service.py — Event loop conflict in search_direct()](#search_servicepy--event-loop-conflict-in-search_direct)
  - [pgvector_backend.py — Wrong index parameter for IVFFlat index](#pgvector_backendpy--wrong-index-parameter-for-ivfflat-index)
  - [web_service.py — HTTPException caught by its own broad except handler](#web_servicepy--httpexception-caught-by-its-own-broad-except-handler)
  - [web_service.py — stat() crash on broken symlinks](#web_servicepy--stat-crash-on-broken-symlinks)
  - [web_mixin.py — Double prefix when route is "/"](#web_mixinpy--double-prefix-when-route-is-)
  - [web_mixin.py — Azure Functions response printed to stdout](#web_mixinpy--azure-functions-response-printed-to-stdout)
  - [auth_handler.py — HTTPBasic auto_error blocks optional auth](#auth_handlerpy--httpbasic-auto_error-blocks-optional-auth)
  - [survey.py — Dict passed to SwaigFunctionResult instead of string](#surveypy--dict-passed-to-swaigfunctionresult-instead-of-string)
  - [concierge.py — Case-sensitive service lookup after lowercasing input](#conciergepy--case-sensitive-service-lookup-after-lowercasing-input)
  - [mcp_gateway/skill.py — Race condition on self.session_id](#mcp_gatewayskillpy--race-condition-on-selfsession_id)
  - [mcp_gateway/skill.py — response.json() called on non-JSON error responses](#mcp_gatewayskillpy--responsejson-called-on-non-json-error-responses)
  - [contexts.py — Missing step-level valid_contexts validation](#contextspy--missing-step-level-valid_contexts-validation)
  - [wikipedia_search/skill.py — String concatenation instead of join](#wikipedia_searchskillpy--string-concatenation-instead-of-join)
  - [datasphere_serverless/skill.py — Distance parameter not sent to API](#datasphere_serverlessskillpy--distance-parameter-not-sent-to-api)
  - [datasphere/skill.py — requests.Session never closed](#datasphereskillpy--requestssession-never-closed)
  - [build_search.py — Wrong attribute name for similarity threshold](#build_searchpy--wrong-attribute-name-for-similarity-threshold)
  - [build_search.py — model_name not passed to IndexBuilder in JSON export](#build_searchpy--model_name-not-passed-to-indexbuilder-in-json-export)
  - [build_search.py — File handle leak in chunk counting](#build_searchpy--file-handle-leak-in-chunk-counting)
  - [gateway_service.py — Shutdown cleanup skipped on signal re-entry](#gateway_servicepy--shutdown-cleanup-skipped-on-signal-re-entry)
  - [mcp_manager.py — Stale client references after stop()](#mcp_managerpy--stale-client-references-after-stop)
  - [query_processor.py — Threading lock lazy init is itself a race condition](#query_processorpy--threading-lock-lazy-init-is-itself-a-race-condition)
  - [query_processor.py — set_global_model fails to find model name attribute](#query_processorpy--set_global_model-fails-to-find-model-name-attribute)
  - [document_processor.py — New SentenceTransformer created on every semantic chunk](#document_processorpy--new-sentencetransformer-created-on-every-semantic-chunk)
  - [document_processor.py — Duplicate context sentences in QA chunking](#document_processorpy--duplicate-context-sentences-in-qa-chunking)
  - [document_processor.py — chunk_size compared against character count instead of word count](#document_processorpy--chunk_size-compared-against-character-count-instead-of-word-count)
  - [native_vector_search/skill.py — "basic" NLP backend rejected as invalid](#native_vector_searchskillpy--basic-nlp-backend-rejected-as-invalid)
  - [registry.py — Path separator hardcoded as colon](#registrypy--path-separator-hardcoded-as-colon)
  - [spider/skill.py — cache.clear() called on None](#spiderskillpy--cacheclear-called-on-none)
- [Low Severity](#low-severity)
  - [Bare except clauses replaced with typed exceptions](#bare-except-clauses-replaced-with-typed-exceptions)

---

## Critical / High Severity

### swml_service.py — Auth bypass: HTTPException returned instead of raised

**File:** `signalwire_agents/core/swml_service.py:662`

**Bug:** When basic auth fails, the code does `return HTTPException(...)` instead of `raise HTTPException(...)`. In FastAPI, returning an exception object does not trigger the HTTP error response — it just returns the exception object as a successful 200 response. This means **authentication was never enforced**; every request was allowed through regardless of credentials.

**Fix:** Changed `return` to `raise`.

```python
# Before (broken — returns exception as data, request succeeds)
return HTTPException(status_code=401, detail="Unauthorized")

# After (correct — raises exception, returns 401 to client)
raise HTTPException(status_code=401, detail="Unauthorized")
```

---

### web_service.py — Path traversal in static file serving

**File:** `signalwire_agents/web/web_service.py:397`

**Bug:** The path traversal check used `str(full_path).startswith(str(dir_path))` which can be bypassed by crafting a directory name that is a prefix of the allowed directory. For example, if the allowed directory is `/srv/static`, a path like `/srv/static_evil/secret.txt` would pass the check because `"/srv/static_evil"` starts with `"/srv/static"`.

**Fix:** Appended `os.sep` to the directory path in the comparison, and added an equality check for the directory itself.

```python
# Before (bypassable)
if not str(full_path).startswith(str(dir_path)):

# After (correct)
if not str(full_path).startswith(str(dir_path) + os.sep) and full_path != dir_path:
```

---

### agent_server.py — Path traversal in static file serving

**File:** `signalwire_agents/agent_server.py:720`

**Bug:** Same path traversal vulnerability as `web_service.py` above — prefix-based directory check could be bypassed with sibling directory names.

**Fix:** Same approach: appended `os.sep` and added equality check.

```python
# Before
if not str(full_path).startswith(str(static_dir)):

# After
if not str(full_path).startswith(str(static_dir) + os.sep) and full_path != static_dir:
```

---

### web_service.py — XSS in directory listing

**File:** `signalwire_agents/web/web_service.py:254-291`

**Bug:** File names and the URL path were interpolated directly into HTML without escaping. A file named `<img src=x onerror=alert(1)>` would execute JavaScript in the browser when viewing the directory listing.

**Fix:** Applied `html.escape()` to all user-controlled values rendered into HTML: file names in links and the URL path in the page title and heading.

```python
# Before (XSS)
items.append(f'<li>📁 <a href="{item.name}/">{item.name}/</a></li>')
f"<title>Directory listing for {url_path}</title>"

# After (escaped)
safe_name = escape(item.name, quote=True)
items.append(f'<li>📁 <a href="{safe_name}/">{safe_name}/</a></li>')
f"<title>Directory listing for {__import__('html').escape(url_path)}</title>"
```

---

### search_engine.py — SQL injection in metadata search

**File:** `signalwire_agents/search/search_engine.py:668-758`

**Bug:** User-supplied search terms were interpolated directly into SQL queries using f-strings in two places: `metadata_text LIKE '%{term}%'` and `content LIKE '%"{term}%'`. An attacker could inject arbitrary SQL by including SQL syntax in their search query.

**Fix:** Replaced all f-string interpolation with parameterized queries using `?` placeholders.

```python
# Before (SQL injection)
conditions.append(f"metadata_text LIKE '%{term}%'")
cursor.execute(query_sql, (count * 10,))

# After (parameterized)
conditions.append("metadata_text LIKE ?")
params.append(f"%{term}%")
cursor.execute(query_sql, params)
```

---

### pgvector_backend.py — SQL injection via collection name

**File:** `signalwire_agents/search/pgvector_backend.py:90,349`

**Bug:** The `collection_name` parameter was used directly in table names without sanitization: `f"chunks_{collection_name}"`. A malicious collection name like `foo; DROP TABLE users;--` could execute arbitrary SQL.

**Fix:** Added regex sanitization to strip all characters except alphanumeric and underscore.

```python
# Before
table_name = f"chunks_{collection_name}"

# After
sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name)
table_name = f"chunks_{sanitized_name}"
```

---

### auth_handler.py — Timing attack on credential comparison

**File:** `signalwire_agents/core/auth_handler.py:189-191`

**Bug:** Basic auth credentials were compared using `==`, which short-circuits on the first differing character. An attacker can measure response times to determine credentials character by character.

**Fix:** Replaced with `secrets.compare_digest()` which runs in constant time regardless of where the strings differ.

```python
# Before (timing-vulnerable)
if auth.username == basic_config['username'] and auth.password == basic_config['password']:

# After (constant-time)
if secrets.compare_digest(auth.username, basic_config['username']) and \
   secrets.compare_digest(auth.password, basic_config['password']):
```

---

### gateway_service.py — Timing attack on token comparison

**File:** `signalwire_agents/mcp_gateway/gateway_service.py:284-541`

**Bug:** Bearer token and basic auth credentials were compared using `==`, making them vulnerable to timing attacks.

**Fix:** Replaced with `hmac.compare_digest()` for constant-time comparison.

```python
# Before
if expected_token and token == expected_token:
if auth.username == server_config.get('auth_user') and auth.password == server_config.get('auth_password'):

# After
if expected_token and hmac.compare_digest(token, expected_token):
if (hmac.compare_digest(auth.username, expected_user) and hmac.compare_digest(auth.password, expected_pass)):
```

---

### auth_mixin.py — Timing attack in GCF/Azure auth

**File:** `signalwire_agents/core/mixins/auth_mixin.py:208,256`

**Bug:** The Google Cloud Function and Azure Function auth handlers compared credentials using direct `==` comparison instead of using the existing `validate_basic_auth()` method which uses timing-safe comparison.

**Fix:** Replaced inline comparison with `self.validate_basic_auth()`.

```python
# Before (timing-vulnerable, duplicated logic)
expected_username, expected_password = self.get_basic_auth_credentials()
return (provided_username == expected_username and provided_password == expected_password)

# After (uses existing timing-safe method)
return self.validate_basic_auth(provided_username, provided_password)
```

---

### web_mixin.py — Security token bypass for all SWAIG functions

**File:** `signalwire_agents/core/mixins/web_mixin.py:703-433`

**Bug:** The security token validation code was commented with "SECURITY BYPASS FOR DEBUGGING" and intentionally never rejected requests with invalid tokens. Invalid tokens were logged but the request was allowed through, meaning **all SWAIG function calls were accessible without valid tokens**.

**Fix:** Replaced the debug bypass with actual enforcement. When a token is invalid and the function has `secure: true` (the default), the request is now rejected with a 401 response.

```python
# Before (bypass — all requests allowed)
req_log.warning("token_invalid")
# Log but continue anyway for debugging

# After (enforced)
req_log.warning("token_invalid")
func_entry = self._tool_registry._swaig_functions.get(function_name)
if func_entry and func_entry.get('secure', True):
    return JSONResponse(status_code=401, content={"error": "Invalid security token"})
```

---

### security_config.py — Password regenerated on every call

**File:** `signalwire_agents/core/security_config.py:255`

**Bug:** When no explicit password was configured, `get_basic_auth_credentials()` called `secrets.token_urlsafe(32)` on every invocation, generating a new random password each time. This meant the password returned during auth setup was different from the one used during validation — making auth unpredictable and potentially broken.

**Fix:** Cache the generated password on the instance so it remains stable across calls.

```python
# Before (new password every call)
password = self.basic_auth_password or secrets.token_urlsafe(32)

# After (generated once, cached)
if not self.basic_auth_password:
    self.basic_auth_password = secrets.token_urlsafe(32)
password = self.basic_auth_password
```

---

### search_service.py — Credential exposure in health endpoint

**File:** `signalwire_agents/search/search_service.py:263`

**Bug:** The `/health` endpoint returned the raw `connection_string` which typically contains database credentials (username, password, host). Anyone who can reach the health endpoint can extract database credentials.

**Fix:** Replaced the connection string value with `"***"`.

```python
# Before (leaks credentials)
"connection_string": self.connection_string if self.backend == 'pgvector' else None

# After (masked)
"connection_string": "***" if self.backend == 'pgvector' and self.connection_string else None
```

---

## Medium Severity

### swml_service.py — Auth parsing fails on passwords containing colons

**File:** `signalwire_agents/core/swml_service.py:908`

**Bug:** Basic auth parsing used `decoded.split(":")` which splits on every colon. Per RFC 7617, the password field may contain colons. A password like `my:secret:pass` would be split into `["my", "secret", "pass"]`, causing a `ValueError` on the tuple unpacking.

**Fix:** Changed to `split(":", 1)` to split only on the first colon.

```python
# Before (breaks on passwords with colons)
username, password = decoded.split(":")

# After (RFC 7617 compliant)
username, password = decoded.split(":", 1)
```

---

### swml_service.py — Port 0 and empty host rejected as falsy

**File:** `signalwire_agents/core/swml_service.py:840-841`

**Bug:** `host = host or self.host` and `port = port or self.port` treat `0` and `""` as falsy. Port `0` is a valid value (meaning "let the OS pick a port"), and `host=""` means "bind all interfaces" (`0.0.0.0`). Both would be silently replaced with the defaults.

**Fix:** Changed to explicit `None` checks.

```python
# Before (rejects port=0 and host="")
host = host or self.host
port = port or self.port

# After (only replaces when explicitly None)
host = host if host is not None else self.host
port = port if port is not None else self.port
```

---

### swml_service.py — URL parameter values not encoded

**File:** `signalwire_agents/core/swml_service.py:1058`

**Bug:** URL query parameters were built using raw f-string interpolation: `f"{k}={v}"`. Values containing `&`, `=`, spaces, or other special characters would corrupt the URL structure.

**Fix:** Replaced with `urllib.parse.urlencode()` which properly percent-encodes all values.

```python
# Before (breaks on special characters)
params = "&".join([f"{k}={v}" for k, v in filtered_params.items()])
url = f"{url}?{params}"

# After (properly encoded)
from urllib.parse import urlencode
url = f"{url}?{urlencode(filtered_params)}"
```

---

### swml_service.py — Document mutation across requests

**File:** `signalwire_agents/core/swml_service.py:706`

**Bug:** `document = self.get_document()` returns a reference to the shared document object. When modifications are applied to this reference, they persist and affect all subsequent requests, causing state leakage between calls.

**Fix:** Deep-copy the document before applying per-request modifications.

```python
# Before (mutates shared state)
document = self.get_document()

# After (isolated per-request copy)
import copy
document = copy.deepcopy(self.get_document())
```

---

### agent_base.py — Operator precedence bug in auto_answer check

**File:** `signalwire_agents/core/agent_base.py:459`

**Bug:** `if not config.get("auto_answer") is False:` is parsed as `if (not config.get("auto_answer")) is False:` due to Python operator precedence. The `not` operator binds tighter than `is`, so this evaluates `not value` first (a boolean), then checks if that boolean `is False`. This produces incorrect results for most values.

**Fix:** Changed to `is not False` which is a single operator with the correct semantics.

```python
# Before (wrong precedence: `not x` then `is False`)
if not config.get("auto_answer") is False:

# After (correct: checks if value is not False)
if config.get("auto_answer") is not False:
```

---

### agent_base.py — URL parameter values not encoded

**File:** `signalwire_agents/core/agent_base.py:1168`

**Bug:** Same as the `swml_service.py` URL encoding bug — query parameters built with raw f-strings.

**Fix:** Same approach: replaced with `urlencode()`.

---

### function_result.py — Falsy response treated as empty

**File:** `signalwire_agents/core/function_result.py:80`

**Bug:** `self.response = response or ""` converts any falsy response to empty string. A response of `0`, `False`, or `[]` would be silently discarded. These are legitimate response values.

**Fix:** Changed to explicit `None` check.

```python
# Before (discards 0, False, [], etc.)
self.response = response or ""

# After (only replaces None)
self.response = response if response is not None else ""
```

---

### function_result.py — TypeError when executing SWML with string content

**File:** `signalwire_agents/core/function_result.py:359`

**Bug:** When `execute_swml()` receives a string and `transfer=True`, it tries to do `action["transfer"] = "true"` on a string, which raises `TypeError: 'str' object does not support item assignment`.

**Fix:** Parse string content to dict via `json.loads()` so the transfer key can be added. Falls back to wrapping in a dict if parsing fails.

```python
# Before (crashes on string + transfer=True)
swml_data = swml_content  # string

# After (parses to dict)
try:
    swml_data = json.loads(swml_content)
except (json.JSONDecodeError, ValueError):
    swml_data = {"raw_swml": swml_content}
```

---

### function_result.py — Caller dict mutation in execute_swml

**File:** `signalwire_agents/core/function_result.py:365`

**Bug:** When `execute_swml()` receives a dict, it used the dict directly: `swml_data = swml_content`. Adding the `transfer` key modifies the caller's original dictionary, which is unexpected and can cause bugs in calling code that reuses the dict.

**Fix:** Shallow-copy the dict before modification.

```python
# Before (mutates caller's dict)
swml_data = swml_content

# After (isolated copy)
swml_data = dict(swml_content)
```

---

### data_map.py — "required" key collision with parameter definitions

**File:** `signalwire_agents/core/data_map.py:131-136,364-365`

**Bug:** Required parameter names were stored in `self._parameters["required"]` — the same dictionary that holds parameter definitions. If a user defines a parameter named `"required"`, it would overwrite the required-params list, or vice versa. This silently corrupts the parameter schema.

**Fix:** Changed the internal key to `"_required"` to avoid collision with user parameter names. The key is converted back to `"required"` when generating the output schema.

```python
# Before (collides with parameter named "required")
self._parameters["required"] = []

# After (internal-only key, no collision)
self._parameters["_required"] = []
```

---

### data_map.py — Incorrect type hint on foreach()

**File:** `signalwire_agents/core/data_map.py:254`

**Bug:** The `foreach()` method had type hint `Union[str, Dict[str, Any]]` but only accepts dict configuration. The `str` type was never valid and misleads callers.

**Fix:** Changed to `Dict[str, Any]`.

---

### search_engine.py — FTS5 query escaping uses invalid backslash syntax

**File:** `signalwire_agents/search/search_engine.py:309-383`

**Bug:** The `_escape_fts_query()` method used backslash escaping (`\(`, `\"`, etc.) for FTS5 special characters. SQLite FTS5 does not support backslash escaping — it uses double-quote wrapping to escape terms. The backslash approach fails silently, producing either no results or incorrect results.

**Fix:** Replaced with proper FTS5 escaping: remove existing double quotes, split into terms, and wrap each term in double quotes.

```python
# Before (invalid FTS5 syntax)
escaped = escaped.replace(char, f'\\{char}')

# After (correct FTS5 quoting)
cleaned = query.replace('"', '')
terms = cleaned.split()
escaped_terms = [f'"{term}"' for term in terms if term]
return ' '.join(escaped_terms)
```

---

### search_engine.py — Boost multiplier not applied to final_score

**File:** `signalwire_agents/search/search_engine.py:508-530`

**Bug:** `_boost_exact_matches()` multiplied `result['score']` for exact matches but never updated `result['final_score']`. Since downstream code uses `final_score` for ranking, the boosts had no effect on the actual result ordering.

**Fix:** Also update `final_score` when applying boost multipliers.

```python
# Before (boost has no effect on ranking)
result['score'] *= 2.0

# After (boost affects ranking)
result['score'] *= 2.0
result['final_score'] = result.get('final_score', result['score']) * 2.0
```

---

### search_engine.py — fetchall() called without prior execute()

**File:** `signalwire_agents/search/search_engine.py:758`

**Bug:** In `_metadata_search()`, `cursor.fetchall()` was called outside the `if specific_conditions:` block. When `specific_conditions` is empty, no query was executed, so `fetchall()` either returns stale results from a previous query or raises an error.

**Fix:** Moved `fetchall()` inside the conditional block and added `else: rows = []`.

```python
# Before (fetchall without execute when no conditions)
if specific_conditions:
    cursor.execute(query_sql, ...)
rows = cursor.fetchall()

# After
if specific_conditions:
    cursor.execute(query_sql, ...)
    rows = cursor.fetchall()
else:
    rows = []
```

---

### search_engine.py — SQLite connection leaks on exceptions

**File:** `signalwire_agents/search/search_engine.py` (7 methods)

**Bug:** Seven methods (`_load_config`, `_vector_search`, `_keyword_search`, `_fallback_search`, `_filename_search`, `_metadata_search`, `_add_vector_scores_to_candidates`) opened SQLite connections with `conn = sqlite3.connect(...)` and closed them with `conn.close()` in the happy path, but if any exception occurred before the close call, the connection leaked. Over time this exhausts file descriptors and SQLite connections.

**Fix:** Added `conn = None` initialization before the try block and `finally: if conn: conn.close()` to ensure cleanup on all code paths.

```python
# Before (leaks on exception)
conn = sqlite3.connect(self.index_path)
# ... work ...
conn.close()

# After (always closes)
conn = None
try:
    conn = sqlite3.connect(self.index_path)
    # ... work ...
finally:
    if conn:
        conn.close()
```

---

### search_service.py — Wrong parameter name passed to SearchRequest

**File:** `signalwire_agents/search/search_service.py:475`

**Bug:** `search_direct()` passed `distance=distance` to the `SearchRequest` constructor, but the `SearchRequest` model has a field named `similarity_threshold`, not `distance`. The value was silently ignored and the default threshold was always used.

**Fix:** Changed to `similarity_threshold=distance`.

```python
# Before (parameter ignored)
SearchRequest(query=query, distance=distance, ...)

# After (parameter applied)
SearchRequest(query=query, similarity_threshold=distance, ...)
```

---

### search_service.py — Event loop conflict in search_direct()

**File:** `signalwire_agents/search/search_service.py:480`

**Bug:** `search_direct()` used `asyncio.get_event_loop()` and `loop.run_until_complete()`. If called from within an already-running async context (e.g., inside a FastAPI handler), this raises `RuntimeError: This event loop is already running`. The fallback created a new event loop and set it globally, which can cause issues in multi-threaded environments.

**Fix:** Check for a running loop first. If one exists, use a ThreadPoolExecutor to run the async method in a separate thread. If no loop is running, use `asyncio.run()` which properly creates and cleans up an event loop.

```python
# Before (crashes in async context)
loop = asyncio.get_event_loop()
response = loop.run_until_complete(self._handle_search(request))

# After (works in both sync and async contexts)
try:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, self._handle_search(request))
        response = future.result()
except RuntimeError:
    response = asyncio.run(self._handle_search(request))
```

---

### pgvector_backend.py — Wrong index parameter for IVFFlat index

**File:** `signalwire_agents/search/pgvector_backend.py:477`

**Bug:** The code sets `SET LOCAL hnsw.ef_search = ...` but the index created elsewhere uses IVFFlat, not HNSW. The `hnsw.ef_search` parameter has no effect on IVFFlat indexes, meaning the search quality tuning was silently ignored.

**Fix:** Changed to `SET LOCAL ivfflat.probes = ...` which is the correct parameter for IVFFlat indexes.

```python
# Before (wrong index type parameter)
cursor.execute(f"SET LOCAL hnsw.ef_search = {max(count, 40)}")

# After (correct for IVFFlat)
cursor.execute(f"SET LOCAL ivfflat.probes = {max(count, 10)}")
```

---

### web_service.py — HTTPException caught by its own broad except handler

**File:** `signalwire_agents/web/web_service.py:401`

**Bug:** The path traversal check raises `HTTPException(status_code=403)` on access denial, but this is immediately caught by the `except Exception:` block below it (since `HTTPException` inherits from `Exception`). The except block raises a different `HTTPException(status_code=403, detail="Invalid path")` — masking the original error detail and making debugging harder.

**Fix:** Added `except HTTPException: raise` before the broad `except Exception:` to let HTTPExceptions pass through.

```python
# Before (HTTPException caught by own handler)
try:
    ...
    raise HTTPException(status_code=403, detail="Access denied")
except Exception:
    raise HTTPException(status_code=403, detail="Invalid path")

# After (HTTPException passes through)
try:
    ...
    raise HTTPException(status_code=403, detail="Access denied")
except HTTPException:
    raise
except Exception:
    raise HTTPException(status_code=403, detail="Invalid path")
```

---

### web_service.py — stat() crash on broken symlinks

**File:** `signalwire_agents/web/web_service.py:217`

**Bug:** `_is_file_allowed()` calls `file_path.stat()` which raises `FileNotFoundError` on broken symlinks or `OSError` on permission issues. This would crash the file-serving handler with an unhandled exception instead of gracefully rejecting the file.

**Fix:** Wrapped `stat()` in a try/except that returns `False` on error.

```python
# Before (crashes on broken symlinks)
if file_path.stat().st_size > self.max_file_size:
    return False

# After (graceful handling)
try:
    if file_path.stat().st_size > self.max_file_size:
        return False
except (OSError, FileNotFoundError):
    return False
```

---

### web_mixin.py — Double prefix when route is "/"

**File:** `signalwire_agents/core/mixins/web_mixin.py:90`

**Bug:** `app.include_router(router, prefix=self.route)` always passes the route as a prefix. When `self.route = "/"`, FastAPI prepends `/` to all routes, resulting in double-slash paths like `//swml` and `//swaig`. These paths may not match client requests.

**Fix:** Only pass prefix when route is not `/`.

```python
# Before (creates //swml when route is "/")
app.include_router(router, prefix=self.route)

# After
if self.route == "/":
    app.include_router(router)
else:
    app.include_router(router, prefix=self.route)
```

---

### web_mixin.py — Azure Functions response printed to stdout

**File:** `signalwire_agents/core/mixins/web_mixin.py:301`

**Bug:** Both CGI and Azure Function modes were handled in the same branch: `if mode in ['cgi', 'azure_function']`. This branch called `print(response)` to return the response to stdout. For CGI mode this is correct (that's how CGI works). For Azure Functions, printing to stdout corrupts the function's output — Azure expects the response object to be returned, not printed.

**Fix:** Split into separate branches so only CGI mode prints.

```python
# Before (prints Azure response to stdout)
if mode in ['cgi', 'azure_function']:
    response = self.handle_serverless_request(event, context, mode)
    print(response)
    return response

# After (only CGI prints)
if mode == 'cgi':
    response = self.handle_serverless_request(event, context, mode)
    print(response)
    return response
elif mode == 'azure_function':
    response = self.handle_serverless_request(event, context, mode)
    return response
```

---

### auth_handler.py — HTTPBasic auto_error blocks optional auth

**File:** `signalwire_agents/core/auth_handler.py:45`

**Bug:** `HTTPBasic()` defaults to `auto_error=True`, which means FastAPI automatically returns a 401 response if no basic auth header is present. This blocks endpoints that should support optional authentication (e.g., endpoints that work with or without auth). Meanwhile, `HTTPBearer` was correctly set to `auto_error=False`.

**Fix:** Set `auto_error=False` to match the bearer auth behavior and let the handler decide whether to reject.

```python
# Before (rejects requests without auth header before handler runs)
self.basic_auth = HTTPBasic()

# After (lets handler decide)
self.basic_auth = HTTPBasic(auto_error=False)
```

---

### survey.py — Dict passed to SwaigFunctionResult instead of string

**File:** `signalwire_agents/prefabs/survey.py:309,350`

**Bug:** Two places passed a Python dict to `SwaigFunctionResult()` which expects a string response message. The dict would be serialized as its `repr()` form (e.g., `{'response': 'text', 'valid': True}`), which is not a useful response for an AI agent to read or speak.

**Fix:** Pass the message string directly.

```python
# Before (dict gets repr'd into ugly string)
return SwaigFunctionResult({
    "response": message,
    "valid": valid,
    "question_id": question_id
})

# After (clean message string)
return SwaigFunctionResult(message)
```

---

### concierge.py — Case-sensitive service lookup after lowercasing input

**File:** `signalwire_agents/prefabs/concierge.py:215`

**Bug:** The handler lowercases the user's `service` input earlier in the function, then checks `if service in self.services`. But `self.services` contains the original-case service names. So a user saying "SPA" (lowercased to "spa") would not match "Spa" in the services list.

**Fix:** Compare against lowercased service names.

```python
# Before (case mismatch after lowering input)
if service in self.services:

# After (consistent case comparison)
if service in [s.lower() for s in self.services]:
```

---

### mcp_gateway/skill.py — Race condition on self.session_id

**File:** `signalwire_agents/skills/mcp_gateway/skill.py:277`

**Bug:** The tool handler stored the session ID on `self.session_id` — a shared instance attribute. If two concurrent calls arrive, they overwrite each other's session ID, causing responses to be associated with the wrong call. This is a race condition in any multi-threaded or async context.

**Fix:** Use a local variable `session_id` instead of the instance attribute.

```python
# Before (shared state, race condition)
self.session_id = global_data['mcp_call_id']
request_data = {"session_id": self.session_id, ...}

# After (local, thread-safe)
session_id = global_data['mcp_call_id']
request_data = {"session_id": session_id, ...}
```

---

### mcp_gateway/skill.py — response.json() called on non-JSON error responses

**File:** `signalwire_agents/skills/mcp_gateway/skill.py:318`

**Bug:** When the HTTP response has an error status code, the code called `response.json()` unconditionally. Many error responses (e.g., 502 Bad Gateway from a proxy, or plain-text errors) are not valid JSON, causing a `JSONDecodeError` that masks the actual error.

**Fix:** Wrapped `response.json()` in a try/except with a fallback that uses the raw response text.

```python
# Before (crashes on non-JSON errors)
error_data = response.json()

# After (graceful fallback)
try:
    error_data = response.json()
    error_msg = error_data.get('error', f'HTTP {response.status_code}')
except (ValueError, requests.exceptions.JSONDecodeError):
    error_msg = f'HTTP {response.status_code}: {response.text[:200]}'
```

---

### contexts.py — Missing step-level valid_contexts validation

**File:** `signalwire_agents/core/contexts.py:680`

**Bug:** The `validate()` method checked context-level `valid_contexts` references (ensuring referenced contexts exist) but did not validate step-level `valid_contexts`. A step could reference a non-existent context, which would only fail at runtime during a call.

**Fix:** Added a second validation loop that checks step-level `valid_contexts` references.

```python
# Added validation
for context_name, context in self._contexts.items():
    for step_name, step in context._steps.items():
        if hasattr(step, '_valid_contexts') and step._valid_contexts:
            for valid_context in step._valid_contexts:
                if valid_context not in self._contexts:
                    raise ValueError(
                        f"Step '{step_name}' in context '{context_name}' "
                        f"references unknown context '{valid_context}'"
                    )
```

---

### wikipedia_search/skill.py — String concatenation instead of join

**File:** `signalwire_agents/skills/wikipedia_search/skill.py:174`

**Bug:** `"\n\n" + "="*50 + "\n\n".join(articles)` is parsed as `"\n\n" + ("="*50) + ("\n\n".join(articles))` due to operator precedence. The separator `"="*50` is only prepended once, not placed between articles. The first article gets no separator before it and the rest are joined with just `"\n\n"`.

**Fix:** Wrap the full separator string in parentheses so `.join()` applies to the complete separator.

```python
# Before (separator only at start, not between articles)
return "\n\n" + "="*50 + "\n\n".join(articles)

# After (separator between every article)
return ("\n\n" + "="*50 + "\n\n").join(articles)
```

---

### datasphere_serverless/skill.py — Distance parameter not sent to API

**File:** `signalwire_agents/skills/datasphere_serverless/skill.py:167`

**Bug:** The `distance` parameter (similarity threshold) was configured and stored in `self.distance` during setup, but was never included in the `webhook_params` sent to the DataSphere API. The API would use its own default distance threshold, ignoring the user's configuration.

**Fix:** Added `"distance": self.distance` to `webhook_params`.

```python
# Before (distance not sent)
webhook_params = {
    "document_id": self.document_id,
    "query_string": "${args.query}",
    "count": self.count
}

# After (distance included)
webhook_params = {
    "document_id": self.document_id,
    "query_string": "${args.query}",
    "count": self.count,
    "distance": self.distance
}
```

---

### datasphere/skill.py — requests.Session never closed

**File:** `signalwire_agents/skills/datasphere/skill.py:280`

**Bug:** The `DataSphereSkill` creates a `requests.Session()` during setup but never closes it. Sessions hold open TCP connections and other resources. Over the lifetime of a long-running agent, leaked sessions accumulate.

**Fix:** Added a `cleanup()` method that closes the session when the skill is unloaded.

```python
def cleanup(self) -> None:
    """Clean up resources when skill is unloaded."""
    if hasattr(self, 'session'):
        self.session.close()
```

---

### build_search.py — Wrong attribute name for similarity threshold

**File:** `signalwire_agents/cli/build_search.py:797,875,1122`

**Bug:** Three places referenced `args.similarity_threshold` but the argparse argument is defined as `--distance-threshold`, which creates the attribute `args.distance_threshold`. This would raise `AttributeError` whenever the search or remote commands were used.

**Fix:** Changed all three occurrences to `args.distance_threshold`.

---

### build_search.py — model_name not passed to IndexBuilder in JSON export

**File:** `signalwire_agents/cli/build_search.py:443`

**Bug:** When using `--format json`, the `IndexBuilder` was created without the `model_name` parameter, even though the user may have specified `--model`. This caused the JSON export to always use the default model, ignoring the user's choice. The SQLite export path correctly passed `model_name`.

**Fix:** Added `model_name=args.model` to the `IndexBuilder` constructor call.

---

### build_search.py — File handle leak in chunk counting

**File:** `signalwire_agents/cli/build_search.py:527`

**Bug:** `sum(len(json.load(open(f))['chunks']) for f in chunk_files_created)` opens files inside a generator expression without closing them. In CPython, reference counting usually closes them promptly, but in other Python implementations (PyPy, etc.) or with many files, this leaks file descriptors.

**Fix:** Replaced with an explicit for loop using context managers.

```python
# Before (file handles not explicitly closed)
total_chunks = sum(len(json.load(open(f))['chunks']) for f in chunk_files_created)

# After (properly closed)
total_chunks = 0
for f in chunk_files_created:
    with open(f) as fh:
        total_chunks += len(json.load(fh)['chunks'])
```

---

### gateway_service.py — Shutdown cleanup skipped on signal re-entry

**File:** `signalwire_agents/mcp_gateway/gateway_service.py:519`

**Bug:** The shutdown method used `_shutdown_requested` to prevent double cleanup. However, signal handlers set `_shutdown_requested = True` before calling `shutdown()`. If the signal handler runs first (which is normal), the shutdown method sees the flag and returns immediately without doing any cleanup.

**Fix:** Added a separate `_shutdown_cleanup_done` flag that only the shutdown method sets, so it correctly detects whether cleanup has already been performed regardless of whether a shutdown was requested.

```python
# Before (cleanup skipped because signal sets flag first)
if self._shutdown_requested:
    return
self._shutdown_requested = True

# After (separate flag for cleanup tracking)
if self._shutdown_cleanup_done:
    return
self._shutdown_cleanup_done = True
```

---

### mcp_manager.py — Stale client references after stop()

**File:** `signalwire_agents/mcp_gateway/mcp_manager.py:480`

**Bug:** `get_service_tools()` and `validate_services()` create temporary MCP clients and call `client.stop()`, but never remove the client from `self.clients`. The stopped client objects remain in the dictionary, potentially being reused later in a stopped/invalid state.

**Fix:** Delete the client from `self.clients` after stopping it.

```python
# Before (stopped client stays in dict)
client.stop()

# After (cleaned up)
client.stop()
if client_key in self.clients:
    del self.clients[client_key]
```

---

### query_processor.py — Threading lock lazy init is itself a race condition

**File:** `signalwire_agents/search/query_processor.py:82`

**Bug:** The model cache lock was lazily initialized: `if _model_lock is None: _model_lock = threading.Lock()`. This check-then-set pattern is itself a race condition — two threads could both see `None` and create separate locks, defeating the purpose of the lock entirely.

**Fix:** Initialize the lock at module level where it's guaranteed to execute once during import.

```python
# Before (race condition in lock initialization)
_model_lock = None
def _get_cached_model(...):
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

# After (initialized at module import time)
import threading
_model_lock = threading.Lock()
```

---

### query_processor.py — set_global_model fails to find model name attribute

**File:** `signalwire_agents/search/query_processor.py:85`

**Bug:** `set_global_model()` only checked for `model.model_name` attribute. Some SentenceTransformer versions use `_model_name` or other attribute names. When the attribute doesn't exist, the model is silently not cached, and a new model is loaded from disk on every query.

**Fix:** Check multiple possible attribute names.

```python
# Before (only checks one attribute)
if model and hasattr(model, 'model_name'):
    _model_cache[model.model_name] = model

# After (checks multiple attributes)
if model:
    model_name = getattr(model, 'model_name', None) or getattr(model, '_model_name', None)
    if model_name:
        _model_cache[model_name] = model
```

---

### document_processor.py — New SentenceTransformer created on every semantic chunk

**File:** `signalwire_agents/search/document_processor.py:903`

**Bug:** `_semantic_chunking()` created a new `SentenceTransformer('sentence-transformers/all-mpnet-base-v2')` instance on every call. Loading a transformer model takes several seconds and hundreds of MB of RAM. For a document with many pages, this multiplies the processing time enormously.

**Fix:** Use the existing `_get_cached_model()` function from `query_processor` which caches model instances.

```python
# Before (loads model from disk every call)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# After (uses cached instance)
from signalwire_agents.search.query_processor import _get_cached_model
model = _get_cached_model('sentence-transformers/all-mpnet-base-v2')
```

---

### document_processor.py — Duplicate context sentences in QA chunking

**File:** `signalwire_agents/search/document_processor.py:1074-1097`

**Bug:** In QA-style chunking, adjacent sentences were added as context without checking for duplicates. If sentence N is QA-relevant, sentence N-1 is added as context. Then if sentence N+1 is also QA-relevant, sentence N is added as context (already in `current_chunk`) and sentence N-1 might be added again to `current_context`. This produces duplicate content in the final chunk.

**Fix:** Added duplicate checks before appending context sentences.

```python
# Before (allows duplicates)
if i > 0 and sentences[i-1] not in current_chunk:
    current_context.append(sentences[i-1])

# After (no duplicates)
if i > 0 and sentences[i-1] not in current_chunk and sentences[i-1] not in current_context:
    current_context.append(sentences[i-1])
```

---

### document_processor.py — chunk_size compared against character count instead of word count

**File:** `signalwire_agents/search/document_processor.py:327,421,529`

**Bug:** Three chunking methods compared `len(content)` (character count) or `current_size` (accumulated character count) against `self.chunk_size`, which is defined in words (default: 500 words). Since average English text has ~6 characters per word, the comparison triggers at ~500 characters (~80 words) instead of the intended 500 words. This produces chunks that are roughly 6x smaller than intended.

**Fix:** Multiplied `chunk_size` by 6 in the comparisons to approximate character count.

```python
# Before (triggers at ~500 characters = ~80 words)
if len(page_content) > self.chunk_size:

# After (triggers at ~3000 characters = ~500 words)
if len(page_content) > self.chunk_size * 6:  # chunk_size is in words, ~6 chars/word
```

---

### native_vector_search/skill.py — "basic" NLP backend rejected as invalid

**File:** `signalwire_agents/skills/native_vector_search/skill.py:337`

**Bug:** The NLP backend validation only accepted `'nltk'` and `'spacy'` as valid values, but the codebase also supports a `'basic'` backend (no external NLP dependencies). Users specifying `nlp_backend='basic'` would get a warning and have it silently changed to `'nltk'`, which might not be installed.

**Fix:** Added `'basic'` to the valid backends list.

```python
# Before (rejects valid 'basic' backend)
if self.index_nlp_backend not in ['nltk', 'spacy']:

# After
if self.index_nlp_backend not in ['basic', 'nltk', 'spacy']:
```

---

### registry.py — Path separator hardcoded as colon

**File:** `signalwire_agents/skills/registry.py:55,310`

**Bug:** `os.environ.get('SIGNALWIRE_SKILL_PATHS', '').split(':')` uses `:` as the path separator. On Windows, the path separator is `;` (since `:` is part of drive letters like `C:\`). A path like `C:\skills;D:\more_skills` would be incorrectly split at the colon in `C:`.

**Fix:** Changed to `os.pathsep` which is `:` on Unix and `;` on Windows.

```python
# Before (breaks on Windows)
env_paths = os.environ.get('SIGNALWIRE_SKILL_PATHS', '').split(':')

# After (cross-platform)
env_paths = os.environ.get('SIGNALWIRE_SKILL_PATHS', '').split(os.pathsep)
```

---

### spider/skill.py — cache.clear() called on None

**File:** `signalwire_agents/skills/spider/skill.py:592`

**Bug:** `cleanup()` calls `self.cache.clear()` when `self.cache` exists as an attribute. However, the attribute can be set to `None` (e.g., when caching is disabled), and calling `.clear()` on `None` raises `AttributeError`.

**Fix:** Added a `None` check.

```python
# Before (crashes when cache is None)
if hasattr(self, 'cache'):
    self.cache.clear()

# After
if hasattr(self, 'cache') and self.cache is not None:
    self.cache.clear()
```

---

## Low Severity

### Bare except clauses replaced with typed exceptions

**Files and locations:**
- `signalwire_agents/agent_server.py:387,410,479` — `except:` → `except (json.JSONDecodeError, ValueError, UnicodeDecodeError):` for JSON parsing
- `signalwire_agents/core/agent_base.py:1202` — `except:` → `except (json.JSONDecodeError, ValueError, TypeError):` for JSON parsing
- `signalwire_agents/core/agent_base.py:1342` — `except:` → `except Exception:` for request body parsing
- `signalwire_agents/core/mixins/prompt_mixin.py:333` — `except:` → `except (ValueError, json.JSONDecodeError):` for JSON parsing
- `signalwire_agents/search/search_engine.py:670` — `except:` → `except Exception:` for metadata column check
- `signalwire_agents/cli/build_search.py:759` — `except:` → `except (ValueError, IndexError):` for count parsing
- `signalwire_agents/cli/build_search.py:1187,1195` — `except:` → `except Exception:` for error response parsing
- `signalwire_agents/cli/dokku.py:2110,2235` — `except:` → `except Exception:` for app.json parsing
- `signalwire_agents/mcp_gateway/gateway_service.py:548` — `except:` → `except Exception:` for server shutdown
- `signalwire_agents/skills/native_vector_search/skill.py:783,815` — `except:` → `except Exception:` for stats and cleanup

**Why this matters:** Bare `except:` catches all exceptions including `SystemExit`, `KeyboardInterrupt`, and `GeneratorExit`. This prevents Ctrl+C from working, can interfere with process shutdown, and makes debugging extremely difficult since all errors are silently swallowed. Using typed exceptions ensures only expected errors are caught while letting fatal signals propagate correctly.

---

## Test Fix

### test_data_map.py — Updated to match internal key rename

**File:** `tests/unit/core/test_data_map.py:61-62`

The test checked for `"required"` in `data_map._parameters`, which was changed to `"_required"` as part of the data_map key collision fix. Updated the test assertions to match.
