"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for the build-search command"""
    parser = argparse.ArgumentParser(
        description='Build local search index from documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with directory (defaults to sentence chunking with 5 sentences per chunk)
  sw-search ./docs

  # Multiple directories
  sw-search ./docs ./examples --file-types md,txt,py

  # Individual files
  sw-search README.md ./docs/guide.md ./src/main.py

  # Mixed sources (directories and files)
  sw-search ./docs README.md ./examples specific_file.txt --file-types md,txt,py

  # Sentence-based chunking with custom parameters
  sw-search ./docs \\
    --chunking-strategy sentence \\
    --max-sentences-per-chunk 10 \\
    --split-newlines 2

  # Sliding window chunking
  sw-search ./docs \\
    --chunking-strategy sliding \\
    --chunk-size 100 \\
    --overlap-size 20

  # Paragraph-based chunking
  sw-search ./docs \\
    --chunking-strategy paragraph \\
    --file-types md,txt,rst

  # Page-based chunking (good for PDFs)
  sw-search ./docs \\
    --chunking-strategy page \\
    --file-types pdf

  # Semantic chunking (groups semantically similar sentences)
  sw-search ./docs \\
    --chunking-strategy semantic \\
    --semantic-threshold 0.6

  # Topic-based chunking (groups by topic changes)
  sw-search ./docs \\
    --chunking-strategy topic \\
    --topic-threshold 0.2

  # QA-optimized chunking (optimized for question-answering)
  sw-search ./docs \\
    --chunking-strategy qa


  # JSON-based chunking (pre-chunked content)
  sw-search ./api_chunks.json \
    --chunking-strategy json \
    --file-types json
  # Full configuration example
  sw-search ./docs ./examples README.md \\
    --output ./knowledge.swsearch \\
    --chunking-strategy sentence \\
    --max-sentences-per-chunk 8 \\
    --file-types md,txt,rst,py \\
    --exclude "**/test/**,**/__pycache__/**" \\
    --languages en,es,fr \\
    --model sentence-transformers/all-mpnet-base-v2 \\
    --tags documentation,api \\
    --verbose

  # Validate an existing index
  sw-search validate ./docs.swsearch

  # Search within an index
  sw-search search ./docs.swsearch "how to create an agent"
  sw-search search ./docs.swsearch "API reference" --count 3 --verbose
  sw-search search ./docs.swsearch "configuration" --tags documentation --json

  # Search via remote API
  sw-search remote http://localhost:8001 "how to create an agent" --index-name docs
  sw-search remote localhost:8001 "API reference" --index-name docs --count 3 --verbose

  # PostgreSQL pgvector backend
  sw-search ./docs \\
    --backend pgvector \\
    --connection-string "postgresql://user:pass@localhost/knowledge" \\
    --output docs_collection

  # Overwrite existing pgvector collection
  sw-search ./docs \\
    --backend pgvector \\
    --connection-string "postgresql://user:pass@localhost/knowledge" \\
    --output docs_collection \\
    --overwrite

  # Search in pgvector collection
  sw-search search docs_collection "how to create an agent" \\
    --backend pgvector \\
    --connection-string "postgresql://user:pass@localhost/knowledge"
        """
    )
    
    parser.add_argument(
        'sources', 
        nargs='+',
        help='Source files and/or directories to index'
    )
    
    parser.add_argument(
        '--output', 
        help='Output .swsearch file (default: sources.swsearch) or collection name for pgvector'
    )
    
    parser.add_argument(
        '--backend',
        choices=['sqlite', 'pgvector'],
        default='sqlite',
        help='Storage backend to use (default: sqlite)'
    )
    
    parser.add_argument(
        '--connection-string',
        help='PostgreSQL connection string for pgvector backend'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing collection (pgvector backend only)'
    )
    
    parser.add_argument(
        '--chunking-strategy',
        choices=['sentence', 'sliding', 'paragraph', 'page', 'semantic', 'topic', 'qa', 'json'],
        default='sentence',
        help='Chunking strategy to use (default: sentence)'
    )
    
    parser.add_argument(
        '--max-sentences-per-chunk',
        type=int,
        default=5,
        help='Maximum sentences per chunk for sentence strategy (default: 5)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50,
        help='Chunk size in words for sliding window strategy (default: 50)'
    )
    
    parser.add_argument(
        '--overlap-size',
        type=int,
        default=10,
        help='Overlap size in words for sliding window strategy (default: 10)'
    )
    
    parser.add_argument(
        '--split-newlines',
        type=int,
        help='Split on multiple newlines (for sentence strategy)'
    )
    
    parser.add_argument(
        '--file-types', 
        default='md,txt,rst',
        help='Comma-separated file extensions to include for directories (default: md,txt,rst)'
    )
    
    parser.add_argument(
        '--exclude', 
        help='Comma-separated glob patterns to exclude (e.g., "**/test/**,**/__pycache__/**")'
    )
    
    parser.add_argument(
        '--languages', 
        default='en',
        help='Comma-separated language codes (default: en)'
    )
    
    parser.add_argument(
        '--model', 
        default='sentence-transformers/all-mpnet-base-v2',
        help='Sentence transformer model name (default: sentence-transformers/all-mpnet-base-v2)'
    )
    
    parser.add_argument(
        '--tags', 
        help='Comma-separated tags to add to all chunks'
    )
    
    parser.add_argument(
        '--index-nlp-backend',
        choices=['nltk', 'spacy'],
        default='nltk',
        help='NLP backend for document processing: nltk (fast, default) or spacy (better quality, slower)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the created index after building'
    )
    
    parser.add_argument(
        '--semantic-threshold',
        type=float,
        default=0.5,
        help='Similarity threshold for semantic chunking (default: 0.5)'
    )
    
    parser.add_argument(
        '--topic-threshold',
        type=float,
        default=0.3,
        help='Similarity threshold for topic chunking (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Validate sources
    valid_sources = []
    for source in args.sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"Warning: Source does not exist, skipping: {source}")
            continue
        valid_sources.append(source_path)
    
    if not valid_sources:
        print("Error: No valid sources found")
        sys.exit(1)
    
    # Validate backend configuration
    if args.backend == 'pgvector' and not args.connection_string:
        print("Error: --connection-string is required for pgvector backend")
        sys.exit(1)
    
    # Default output filename
    if not args.output:
        if args.backend == 'sqlite':
            if len(valid_sources) == 1:
                # Single source - use its name
                source_name = valid_sources[0].stem if valid_sources[0].is_file() else valid_sources[0].name
                args.output = f"{source_name}.swsearch"
            else:
                # Multiple sources - use generic name
                args.output = "sources.swsearch"
        else:
            # For pgvector, use a default collection name
            if len(valid_sources) == 1:
                source_name = valid_sources[0].stem if valid_sources[0].is_file() else valid_sources[0].name
                args.output = source_name
            else:
                args.output = "documents"
    
    # Ensure output has .swsearch extension for sqlite
    if args.backend == 'sqlite' and not args.output.endswith('.swsearch'):
        args.output += '.swsearch'
    
    # Parse lists
    file_types = [ft.strip() for ft in args.file_types.split(',')]
    exclude_patterns = [p.strip() for p in args.exclude.split(',')] if args.exclude else None
    languages = [lang.strip() for lang in args.languages.split(',')]
    tags = [tag.strip() for tag in args.tags.split(',')] if args.tags else None
    
    if args.verbose:
        print(f"Building search index:")
        print(f"  Backend: {args.backend}")
        print(f"  Sources: {[str(s) for s in valid_sources]}")
        if args.backend == 'sqlite':
            print(f"  Output file: {args.output}")
        else:
            print(f"  Collection name: {args.output}")
            print(f"  Connection: {args.connection_string}")
        print(f"  File types (for directories): {file_types}")
        print(f"  Exclude patterns: {exclude_patterns}")
        print(f"  Languages: {languages}")
        print(f"  Model: {args.model}")
        print(f"  Chunking strategy: {args.chunking_strategy}")
        print(f"  Index NLP backend: {args.index_nlp_backend}")
        
        if args.chunking_strategy == 'sentence':
            print(f"  Max sentences per chunk: {args.max_sentences_per_chunk}")
            if args.split_newlines:
                print(f"  Split on newlines: {args.split_newlines}")
        elif args.chunking_strategy == 'sliding':
            print(f"  Chunk size (words): {args.chunk_size}")
            print(f"  Overlap size (words): {args.overlap_size}")
        elif args.chunking_strategy == 'paragraph':
            print(f"  Chunking by paragraphs (double newlines)")
        elif args.chunking_strategy == 'page':
            print(f"  Chunking by pages")
        elif args.chunking_strategy == 'semantic':
            print(f"  Semantic chunking (similarity threshold: {args.semantic_threshold})")
        elif args.chunking_strategy == 'topic':
            print(f"  Topic-based chunking (similarity threshold: {args.topic_threshold})")
        elif args.chunking_strategy == 'qa':
            print(f"  QA-optimized chunking")
        
        print(f"  Tags: {tags}")
        print()
    
    try:
        # Create index builder - import only when actually needed
        from signalwire_agents.search.index_builder import IndexBuilder
        builder = IndexBuilder(
            model_name=args.model,
            chunking_strategy=args.chunking_strategy,
            max_sentences_per_chunk=args.max_sentences_per_chunk,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap_size,
            split_newlines=args.split_newlines,
            index_nlp_backend=args.index_nlp_backend,
            verbose=args.verbose,
            semantic_threshold=args.semantic_threshold,
            topic_threshold=args.topic_threshold,
            backend=args.backend,
            connection_string=args.connection_string
        )
        
        # Build index with multiple sources
        builder.build_index_from_sources(
            sources=valid_sources,
            output_file=args.output,
            file_types=file_types,
            exclude_patterns=exclude_patterns,
            languages=languages,
            tags=tags,
            overwrite=args.overwrite if args.backend == 'pgvector' else False
        )
        
        # Validate if requested
        if args.validate:
            if args.verbose:
                print("\nValidating index...")
            
            validation = builder.validate_index(args.output)
            if validation['valid']:
                print(f"✓ Index validation successful:")
                print(f"  Chunks: {validation['chunk_count']}")
                print(f"  Files: {validation['file_count']}")
                if args.verbose:
                    print(f"  Config: {validation['config']}")
            else:
                print(f"✗ Index validation failed: {validation['error']}")
                sys.exit(1)
        
        if args.backend == 'sqlite':
            print(f"\n✓ Search index created successfully: {args.output}")
        else:
            print(f"\n✓ Search collection created successfully: {args.output}")
            print(f"   Connection: {args.connection_string}")
        
    except KeyboardInterrupt:
        print("\n\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError building index: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def validate_command():
    """Validate an existing search index"""
    parser = argparse.ArgumentParser(description='Validate a search index file')
    parser.add_argument('index_file', help='Path to .swsearch file to validate')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    if not Path(args.index_file).exists():
        print(f"Error: Index file does not exist: {args.index_file}")
        sys.exit(1)
    
    try:
        from signalwire_agents.search.index_builder import IndexBuilder
        builder = IndexBuilder()
        
        validation = builder.validate_index(args.index_file)
        
        if validation['valid']:
            print(f"✓ Index is valid: {args.index_file}")
            print(f"  Chunks: {validation['chunk_count']}")
            print(f"  Files: {validation['file_count']}")
            
            if args.verbose and 'config' in validation:
                print("\nConfiguration:")
                for key, value in validation['config'].items():
                    print(f"  {key}: {value}")
        else:
            print(f"✗ Index validation failed: {validation['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error validating index: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def search_command():
    """Search within an existing search index"""
    parser = argparse.ArgumentParser(description='Search within a .swsearch index file or pgvector collection')
    parser.add_argument('index_source', help='Path to .swsearch file or collection name for pgvector')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--backend', choices=['sqlite', 'pgvector'], default='sqlite',
                       help='Storage backend (default: sqlite)')
    parser.add_argument('--connection-string', help='PostgreSQL connection string for pgvector backend')
    parser.add_argument('--count', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--distance-threshold', type=float, default=0.0, help='Minimum similarity score (default: 0.0)')
    parser.add_argument('--tags', help='Comma-separated tags to filter by')
    parser.add_argument('--query-nlp-backend', choices=['nltk', 'spacy'], default='nltk', 
                       help='NLP backend for query processing: nltk (fast, default) or spacy (better quality, slower)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--no-content', action='store_true', help='Hide content in results (show only metadata)')
    
    args = parser.parse_args()
    
    # Validate backend configuration
    if args.backend == 'pgvector' and not args.connection_string:
        print("Error: --connection-string is required for pgvector backend")
        sys.exit(1)
    
    if args.backend == 'sqlite' and not Path(args.index_source).exists():
        print(f"Error: Index file does not exist: {args.index_source}")
        sys.exit(1)
    
    try:
        # Import search dependencies
        try:
            from signalwire_agents.search.search_engine import SearchEngine
            from signalwire_agents.search.query_processor import preprocess_query
        except ImportError as e:
            print(f"Error: Search functionality not available. Install with: pip install signalwire-agents[search]")
            print(f"Details: {e}")
            sys.exit(1)
        
        # Load search engine
        if args.verbose:
            if args.backend == 'sqlite':
                print(f"Loading search index: {args.index_source}")
            else:
                print(f"Connecting to pgvector collection: {args.index_source}")
        
        if args.backend == 'sqlite':
            engine = SearchEngine(backend='sqlite', index_path=args.index_source)
        else:
            engine = SearchEngine(backend='pgvector', connection_string=args.connection_string,
                                collection_name=args.index_source)
        
        # Get index stats
        stats = engine.get_stats()
        if args.verbose:
            print(f"Index contains {stats['total_chunks']} chunks from {stats['total_files']} files")
            print(f"Searching for: '{args.query}'")
            print(f"Query NLP Backend: {args.query_nlp_backend}")
            print()
        
        # Preprocess query
        enhanced = preprocess_query(args.query, vector=True, query_nlp_backend=args.query_nlp_backend)
        
        # Parse tags if provided
        tags = [tag.strip() for tag in args.tags.split(',')] if args.tags else None
        
        # Perform search
        results = engine.search(
            query_vector=enhanced.get('vector'),
            enhanced_text=enhanced.get('enhanced_text', args.query),
            count=args.count,
            distance_threshold=args.distance_threshold,
            tags=tags
        )
        
        if args.json:
            # Output as JSON
            import json
            output = {
                'query': args.query,
                'enhanced_query': enhanced.get('enhanced_text', args.query),
                'count': len(results),
                'results': []
            }
            
            for i, result in enumerate(results):
                result_data = {
                    'rank': i + 1,
                    'score': result['score'],
                    'metadata': result['metadata']
                }
                if not args.no_content:
                    result_data['content'] = result['content']
                output['results'].append(result_data)
            
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            if not results:
                print(f"No results found for '{args.query}'")
                if tags:
                    print(f"(searched with tags: {tags})")
                sys.exit(0)
            
            print(f"Found {len(results)} result(s) for '{args.query}':")
            if enhanced.get('enhanced_text') != args.query:
                print(f"Enhanced query: '{enhanced.get('enhanced_text')}'")
            if tags:
                print(f"Filtered by tags: {tags}")
            print("=" * 80)
            
            for i, result in enumerate(results):
                print(f"\n[{i+1}] Score: {result['score']:.4f}")
                
                # Show metadata
                metadata = result['metadata']
                print(f"File: {metadata.get('filename', 'Unknown')}")
                if metadata.get('section'):
                    print(f"Section: {metadata['section']}")
                if metadata.get('line_start'):
                    print(f"Lines: {metadata['line_start']}-{metadata.get('line_end', metadata['line_start'])}")
                if metadata.get('tags'):
                    print(f"Tags: {', '.join(metadata['tags'])}")
                
                # Show content unless suppressed
                if not args.no_content:
                    content = result['content']
                    if len(content) > 500 and not args.verbose:
                        content = content[:500] + "..."
                    print(f"\nContent:\n{content}")
                
                if i < len(results) - 1:
                    print("-" * 80)
        
    except Exception as e:
        print(f"Error searching index: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def remote_command():
    """Search via remote API endpoint"""
    parser = argparse.ArgumentParser(description='Search via remote API endpoint')
    parser.add_argument('endpoint', help='Remote API endpoint URL (e.g., http://localhost:8001)')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--index-name', required=True, help='Name of the index to search')
    parser.add_argument('--count', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--distance-threshold', type=float, default=0.0, help='Minimum similarity score (default: 0.0)')
    parser.add_argument('--tags', help='Comma-separated tags to filter by')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--no-content', action='store_true', help='Hide content in results (show only metadata)')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Ensure endpoint starts with http:// or https://
    endpoint = args.endpoint
    if not endpoint.startswith(('http://', 'https://')):
        endpoint = f"http://{endpoint}"
    
    # Ensure endpoint ends with /search
    if not endpoint.endswith('/search'):
        if endpoint.endswith('/'):
            endpoint += 'search'
        else:
            endpoint += '/search'
    
    try:
        import requests
    except ImportError:
        print("Error: requests library not available. Install with: pip install requests")
        sys.exit(1)
    
    # Prepare request payload
    payload = {
        'query': args.query,
        'index_name': args.index_name,
        'count': args.count,
        'distance_threshold': args.distance_threshold
    }
    
    if args.tags:
        payload['tags'] = [tag.strip() for tag in args.tags.split(',')]
    
    if args.verbose:
        print(f"Searching remote endpoint: {endpoint}")
        print(f"Payload: {payload}")
        print()
    
    try:
        # Make the API request
        response = requests.post(
            endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=args.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if args.json:
                # Output raw JSON response
                import json
                print(json.dumps(result, indent=2))
            else:
                # Human-readable output
                results = result.get('results', [])
                if not results:
                    print(f"No results found for '{args.query}' in index '{args.index_name}'")
                    sys.exit(0)
                
                print(f"Found {len(results)} result(s) for '{args.query}' in index '{args.index_name}':")
                if result.get('enhanced_query') and result.get('enhanced_query') != args.query:
                    print(f"Enhanced query: '{result.get('enhanced_query')}'")
                print("=" * 80)
                
                for i, search_result in enumerate(results):
                    print(f"\n[{i+1}] Score: {search_result.get('score', 0):.4f}")
                    
                    # Show metadata
                    metadata = search_result.get('metadata', {})
                    print(f"File: {metadata.get('filename', 'Unknown')}")
                    if metadata.get('section'):
                        print(f"Section: {metadata['section']}")
                    if metadata.get('line_start'):
                        print(f"Lines: {metadata['line_start']}-{metadata.get('line_end', metadata['line_start'])}")
                    if metadata.get('tags'):
                        print(f"Tags: {', '.join(metadata['tags'])}")
                    
                    # Show content unless suppressed
                    if not args.no_content and 'content' in search_result:
                        content = search_result['content']
                        if len(content) > 500 and not args.verbose:
                            content = content[:500] + "..."
                        print(f"\nContent:\n{content}")
                    
                    if i < len(results) - 1:
                        print("-" * 80)
        
        elif response.status_code == 404:
            try:
                error_detail = response.json()
                error_msg = error_detail.get('detail', 'Index not found')
            except:
                error_msg = 'Index not found'
            print(f"Error: {error_msg}")
            sys.exit(1)
        else:
            try:
                error_detail = response.json()
                error_msg = error_detail.get('detail', f'HTTP {response.status_code}')
            except:
                error_msg = f'HTTP {response.status_code}: {response.text}'
            print(f"Error: {error_msg}")
            sys.exit(1)
            
    except requests.ConnectionError:
        print(f"Error: Could not connect to {endpoint}")
        print("Make sure the search server is running")
        sys.exit(1)
    except requests.Timeout:
        print(f"Error: Request timed out after {args.timeout} seconds")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error making request: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def console_entry_point():
    """Console script entry point for pip installation"""
    import sys
    
    # Fast help check - show help without importing heavy modules
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""usage: sw-search [-h] [--output OUTPUT] [--chunking-strategy {sentence,sliding,paragraph,page,semantic,topic,qa}]
                 [--max-sentences-per-chunk MAX_SENTENCES_PER_CHUNK] [--chunk-size CHUNK_SIZE]
                 [--overlap-size OVERLAP_SIZE] [--split-newlines SPLIT_NEWLINES] [--file-types FILE_TYPES]
                 [--exclude EXCLUDE] [--languages LANGUAGES] [--model MODEL] [--tags TAGS]
                 [--index-nlp-backend {nltk,spacy}] [--verbose] [--validate]
                 [--semantic-threshold SEMANTIC_THRESHOLD] [--topic-threshold TOPIC_THRESHOLD]
                 sources [sources ...]

Build local search index from documents

positional arguments:
  sources               Source files and/or directories to index

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Output .swsearch file (default: sources.swsearch)
  --chunking-strategy {sentence,sliding,paragraph,page,semantic,topic,qa}
                        Chunking strategy to use (default: sentence)
  --max-sentences-per-chunk MAX_SENTENCES_PER_CHUNK
                        Maximum sentences per chunk for sentence strategy (default: 5)
  --chunk-size CHUNK_SIZE
                        Chunk size in words for sliding window strategy (default: 50)
  --overlap-size OVERLAP_SIZE
                        Overlap size in words for sliding window strategy (default: 10)
  --split-newlines SPLIT_NEWLINES
                        Split on multiple newlines (for sentence strategy)
  --file-types FILE_TYPES
                        Comma-separated file extensions to include for directories (default: md,txt,rst)
  --exclude EXCLUDE     Comma-separated glob patterns to exclude (e.g., "**/test/**,**/__pycache__/**")
  --languages LANGUAGES
                        Comma-separated language codes (default: en)
  --model MODEL         Sentence transformer model name (default: sentence-transformers/all-mpnet-base-v2)
  --tags TAGS           Comma-separated tags to add to all chunks
  --index-nlp-backend {nltk,spacy}
                        NLP backend for document processing: nltk (fast, default) or spacy (better quality, slower)
  --verbose             Enable verbose output
  --validate            Validate the created index after building
  --semantic-threshold SEMANTIC_THRESHOLD
                        Similarity threshold for semantic chunking (default: 0.5)
  --topic-threshold TOPIC_THRESHOLD
                        Similarity threshold for topic chunking (default: 0.3)

Examples:
  # Basic usage with directory (defaults to sentence chunking with 5 sentences per chunk)
  sw-search ./docs

  # Multiple directories
  sw-search ./docs ./examples --file-types md,txt,py

  # Individual files
  sw-search README.md ./docs/guide.md ./src/main.py

  # Mixed sources (directories and files)
  sw-search ./docs README.md ./examples specific_file.txt --file-types md,txt,py

  # Sentence-based chunking with custom parameters
  sw-search ./docs \\
    --chunking-strategy sentence \\
    --max-sentences-per-chunk 10 \\
    --split-newlines 2

  # Sliding window chunking
  sw-search ./docs \\
    --chunking-strategy sliding \\
    --chunk-size 100 \\
    --overlap-size 20

  # Paragraph-based chunking
  sw-search ./docs \\
    --chunking-strategy paragraph \\
    --file-types md,txt,rst

  # Page-based chunking (good for PDFs)
  sw-search ./docs \\
    --chunking-strategy page \\
    --file-types pdf

  # Semantic chunking (groups semantically similar sentences)
  sw-search ./docs \\
    --chunking-strategy semantic \\
    --semantic-threshold 0.6

  # Topic-based chunking (groups by topic changes)
  sw-search ./docs \\
    --chunking-strategy topic \\
    --topic-threshold 0.2

  # QA-optimized chunking (optimized for question-answering)
  sw-search ./docs \\
    --chunking-strategy qa

  # Full configuration example
  sw-search ./docs ./examples README.md \\
    --output ./knowledge.swsearch \\
    --chunking-strategy sentence \\
    --max-sentences-per-chunk 8 \\
    --file-types md,txt,rst,py \\
    --exclude "**/test/**,**/__pycache__/**" \\
    --languages en,es,fr \\
    --model sentence-transformers/all-mpnet-base-v2 \\
    --tags documentation,api \\
    --verbose

  # Validate an existing index
  sw-search validate ./docs.swsearch

  # Search within an index
  sw-search search ./docs.swsearch "how to create an agent"
  sw-search search ./docs.swsearch "API reference" --count 3 --verbose
  sw-search search ./docs.swsearch "configuration" --tags documentation --json

  # Search via remote API
  sw-search remote http://localhost:8001 "how to create an agent" --index-name docs
  sw-search remote localhost:8001 "API reference" --index-name docs --count 3 --verbose
""")
        return
    
    # Check for subcommands
    if len(sys.argv) > 1:
        if sys.argv[1] == 'validate':
            # Remove 'validate' from argv and call validate_command
            sys.argv.pop(1)
            validate_command()
            return
        elif sys.argv[1] == 'search':
            # Remove 'search' from argv and call search_command
            sys.argv.pop(1)
            search_command()
            return
        elif sys.argv[1] == 'remote':
            # Remove 'remote' from argv and call remote_command
            sys.argv.pop(1)
            remote_command()
            return
    
    # Regular build command
    main()

if __name__ == '__main__':
    main() 