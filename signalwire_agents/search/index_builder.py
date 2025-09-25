"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import fnmatch

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .document_processor import DocumentProcessor
from .query_processor import preprocess_document_content

logger = logging.getLogger(__name__)

class IndexBuilder:
    """Build searchable indexes from document directories"""
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        chunking_strategy: str = 'sentence',
        max_sentences_per_chunk: int = 5,
        chunk_size: int = 50,
        chunk_overlap: int = 10,
        split_newlines: Optional[int] = None,
        index_nlp_backend: str = 'nltk',
        verbose: bool = False,
        semantic_threshold: float = 0.5,
        topic_threshold: float = 0.3,
        backend: str = 'sqlite',
        connection_string: Optional[str] = None
    ):
        """
        Initialize the index builder
        
        Args:
            model_name: Name of the sentence transformer model to use
            chunking_strategy: Strategy for chunking documents ('sentence', 'sliding', 'paragraph', 'page', 'semantic', 'topic', 'qa', 'json')
            max_sentences_per_chunk: For sentence strategy (default: 5)
            chunk_size: For sliding strategy - words per chunk (default: 50)
            chunk_overlap: For sliding strategy - overlap in words (default: 10)
            split_newlines: For sentence strategy - split on multiple newlines (optional)
            index_nlp_backend: NLP backend for indexing (default: 'nltk')
            verbose: Whether to enable verbose logging (default: False)
            semantic_threshold: Similarity threshold for semantic chunking (default: 0.5)
            topic_threshold: Similarity threshold for topic chunking (default: 0.3)
            backend: Storage backend ('sqlite' or 'pgvector') (default: 'sqlite')
            connection_string: PostgreSQL connection string for pgvector backend
        """
        self.model_name = model_name
        self.chunking_strategy = chunking_strategy
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_newlines = split_newlines
        self.index_nlp_backend = index_nlp_backend
        self.verbose = verbose
        self.semantic_threshold = semantic_threshold
        self.topic_threshold = topic_threshold
        self.backend = backend
        self.connection_string = connection_string
        self.model = None
        
        # Validate backend
        if self.backend not in ['sqlite', 'pgvector']:
            raise ValueError(f"Invalid backend '{self.backend}'. Must be 'sqlite' or 'pgvector'")
        
        # Validate NLP backend
        if self.index_nlp_backend not in ['nltk', 'spacy']:
            logger.warning(f"Invalid index_nlp_backend '{self.index_nlp_backend}', using 'nltk'")
            self.index_nlp_backend = 'nltk'
        
        self.doc_processor = DocumentProcessor(
            chunking_strategy=chunking_strategy,
            max_sentences_per_chunk=max_sentences_per_chunk,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_newlines=split_newlines,
            index_nlp_backend=self.index_nlp_backend,
            verbose=self.verbose,
            semantic_threshold=self.semantic_threshold,
            topic_threshold=self.topic_threshold
        )
    
    def _extract_metadata_from_json_content(self, content: str) -> tuple[Dict[str, Any], str]:
        """
        Extract metadata from JSON content if present
        
        Returns:
            (metadata_dict, metadata_text)
        """
        metadata_dict = {}
        
        # Try to extract metadata from JSON structure in content
        if '"metadata":' in content:
            try:
                # Look for metadata object in content
                import re
                # Find all metadata objects
                pattern = r'"metadata"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    try:
                        json_metadata = json.loads(match.group(1))
                        # Merge all found metadata
                        if isinstance(json_metadata, dict):
                            metadata_dict.update(json_metadata)
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Error extracting JSON metadata: {e}")
        
        # Create searchable text from all metadata keys and values
        metadata_text_parts = []
        for key, value in metadata_dict.items():
            # Add key
            metadata_text_parts.append(str(key))
            # Add value(s)
            if isinstance(value, list):
                metadata_text_parts.extend(str(v) for v in value)
            else:
                metadata_text_parts.append(str(value))
        
        metadata_text = ' '.join(metadata_text_parts).lower()
        
        return metadata_dict, metadata_text
    
    def _load_model(self):
        """Load embedding model (lazy loading)"""
        if self.model is None:
            if not SentenceTransformer:
                raise ImportError("sentence-transformers is required for embedding generation. Install with: pip install sentence-transformers")
            
            if self.verbose:
                print(f"Loading embedding model: {self.model_name}")
            
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load model '{self.model_name}': {e}")
                raise
    
    def build_index_from_sources(self, sources: List[Path], output_file: str, 
                                file_types: List[str], exclude_patterns: Optional[List[str]] = None,
                                languages: List[str] = None, tags: Optional[List[str]] = None,
                                overwrite: bool = False):
        """
        Build complete search index from multiple sources (files and directories)
        
        Args:
            sources: List of Path objects (files and/or directories)
            output_file: Output .swsearch file path
            file_types: List of file extensions to include for directories
            exclude_patterns: Glob patterns to exclude
            languages: List of languages to support
            tags: Global tags to add to all chunks
        """
        
        # Discover files from all sources
        files = self._discover_files_from_sources(sources, file_types, exclude_patterns)
        if self.verbose:
            print(f"Found {len(files)} files to process")
        
        if not files:
            print("No files found to process. Check your sources, file types and exclude patterns.")
            return
        
        # Process documents
        chunks = []
        print(f"Processing {len(files)} files...")
        for file_path in files:
            try:
                # For individual files, use the file's parent as the base directory
                # For files from directories, use the original source directory
                base_dir = self._get_base_directory_for_file(file_path, sources)
                file_chunks = self._process_file(file_path, base_dir, tags)
                chunks.extend(file_chunks)
                if self.verbose or file_path.suffix == '.json':
                    print(f"  {file_path}: {len(file_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                if self.verbose:
                    print(f"Error processing {file_path}: {e}")
        
        if not chunks:
            print("No chunks created from documents. Check file contents and processing.")
            return
        
        if self.verbose:
            print(f"Created {len(chunks)} total chunks")
        
        # Generate embeddings
        self._load_model()
        if self.verbose:
            print(f"Generating embeddings for {len(chunks)} chunks...")
        else:
            print(f"Generating embeddings for {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Preprocess content for better search
                processed = preprocess_document_content(
                    chunk['content'], 
                    language=chunk.get('language', 'en'),
                    index_nlp_backend=self.index_nlp_backend
                )
                
                chunk['processed_content'] = processed['enhanced_text']
                
                # Include tags in keywords for better search matching
                keywords = processed.get('keywords', [])
                chunk_tags = chunk.get('tags', [])
                if chunk_tags:
                    # Add tags to keywords list for FTS matching
                    keywords.extend(chunk_tags)
                    # Remove duplicates while preserving order
                    keywords = list(dict.fromkeys(keywords))
                
                chunk['keywords'] = keywords
                
                # For embedding, include tags in the text for better semantic matching
                embedding_text = processed['enhanced_text']
                if chunk_tags:
                    # Append tags to the text for embedding generation
                    embedding_text += " " + " ".join(chunk_tags)
                
                # Generate embedding (suppress progress bar)
                embedding = self.model.encode(embedding_text, show_progress_bar=False)
                chunk['embedding'] = embedding.tobytes()
                
                # Show progress more frequently
                show_every = 50 if len(chunks) > 500 else max(10, len(chunks) // 10)
                if (i + 1) % show_every == 0 or (i + 1) == len(chunks):
                    progress_pct = ((i + 1) / len(chunks)) * 100
                    print(f"  Progress: {i + 1}/{len(chunks)} chunks ({progress_pct:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                # Use original content as fallback
                chunk['processed_content'] = chunk['content']
                chunk['keywords'] = []
                # Create zero embedding as fallback
                if np:
                    embedding = np.zeros(768, dtype=np.float32)
                    chunk['embedding'] = embedding.tobytes()
                else:
                    chunk['embedding'] = b''
        
        # Store chunks based on backend
        if self.backend == 'sqlite':
            # Create SQLite database
            sources_info = [str(s) for s in sources]
            self._create_database(output_file, chunks, languages or ['en'], sources_info, file_types)
            
            if self.verbose:
                print(f"Index created: {output_file}")
                print(f"Total chunks: {len(chunks)}")
        else:
            # Use pgvector backend
            self._store_chunks_pgvector(chunks, output_file, languages or ['en'], overwrite)

    def build_index(self, source_dir: str, output_file: str, 
                   file_types: List[str], exclude_patterns: Optional[List[str]] = None,
                   languages: List[str] = None, tags: Optional[List[str]] = None):
        """
        Build complete search index from a single directory
        
        Args:
            source_dir: Directory to scan for documents
            output_file: Output .swsearch file path
            file_types: List of file extensions to include
            exclude_patterns: Glob patterns to exclude
            languages: List of languages to support
            tags: Global tags to add to all chunks
        """
        
        # Convert to new multi-source method
        sources = [Path(source_dir)]
        self.build_index_from_sources(sources, output_file, file_types, exclude_patterns, languages, tags)

    def _get_base_directory_for_file(self, file_path: Path, sources: List[Path]) -> str:
        """
        Determine the appropriate base directory for a file to calculate relative paths
        
        Args:
            file_path: The file being processed
            sources: List of original source paths
            
        Returns:
            Base directory path as string
        """
        
        # Check if this file was specified directly as a source
        if file_path in sources:
            # For individual files, use the parent directory
            return str(file_path.parent)
        
        # Check if this file is within any of the source directories
        for source in sources:
            if source.is_dir():
                try:
                    # Check if file_path is relative to this source directory
                    file_path.relative_to(source)
                    return str(source)
                except ValueError:
                    # file_path is not relative to this source
                    continue
        
        # Fallback: use the file's parent directory
        return str(file_path.parent)

    def _discover_files_from_sources(self, sources: List[Path], file_types: List[str], 
                                   exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """
        Discover files from multiple sources (files and directories)
        
        Args:
            sources: List of Path objects (files and/or directories)
            file_types: List of file extensions to include for directories
            exclude_patterns: Glob patterns to exclude
            
        Returns:
            List of file paths to process
        """
        
        files = []
        supported_extensions = set(ft.lstrip('.').lower() for ft in file_types)
        
        for source in sources:
            if source.is_file():
                # Individual file - check if it's supported
                file_ext = source.suffix.lstrip('.').lower()
                if file_ext in supported_extensions or not file_ext:  # Allow extensionless files
                    # Check exclusions
                    if self._is_file_excluded(source, exclude_patterns):
                        if self.verbose:
                            print(f"Excluded file: {source}")
                        continue
                    
                    files.append(source)
                    if self.verbose:
                        print(f"Added individual file: {source}")
                else:
                    if self.verbose:
                        print(f"Skipped unsupported file type: {source} (extension: {file_ext})")
                        
            elif source.is_dir():
                # Directory - use existing discovery logic
                dir_files = self._discover_files(str(source), file_types, exclude_patterns)
                files.extend(dir_files)
                if self.verbose:
                    print(f"Added {len(dir_files)} files from directory: {source}")
            else:
                if self.verbose:
                    print(f"Skipped non-existent or invalid source: {source}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        return unique_files

    def _is_file_excluded(self, file_path: Path, exclude_patterns: Optional[List[str]] = None) -> bool:
        """
        Check if a file should be excluded based on exclude patterns
        
        Args:
            file_path: Path to check
            exclude_patterns: List of glob patterns to exclude
            
        Returns:
            True if file should be excluded
        """
        
        if not exclude_patterns:
            return False
        
        import fnmatch
        
        file_str = str(file_path)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                return True
        
        return False

    def _discover_files(self, source_dir: str, file_types: List[str], 
                       exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover files to index"""
        files = []
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
        for file_type in file_types:
            # Clean up file type (remove leading dots)
            clean_type = file_type.lstrip('.')
            pattern = f"**/*.{clean_type}"
            
            for file_path in source_path.glob(pattern):
                # Skip directories
                if not file_path.is_file():
                    continue
                
                # Check exclusions
                if exclude_patterns:
                    excluded = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(str(file_path), pattern):
                            excluded = True
                            break
                    if excluded:
                        if self.verbose:
                            print(f"Excluded: {file_path}")
                        continue
                
                files.append(file_path)
        
        return files
    
    def _process_file(self, file_path: Path, source_dir: str, 
                     global_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process single file into chunks"""
        try:
            relative_path = str(file_path.relative_to(source_dir))
            file_extension = file_path.suffix.lower()
            
            # Handle different file types appropriately
            if file_extension == '.pdf':
                # Use document processor for PDF extraction
                content_result = self.doc_processor._extract_text_from_file(str(file_path))
                if isinstance(content_result, str) and content_result.startswith('{"error"'):
                    if self.verbose:
                        print(f"Skipping PDF file (extraction failed): {file_path}")
                    return []
                content = content_result
            elif file_extension in ['.docx', '.xlsx', '.pptx']:
                # Use document processor for Office documents
                content_result = self.doc_processor._extract_text_from_file(str(file_path))
                if isinstance(content_result, str) and content_result.startswith('{"error"'):
                    if self.verbose:
                        print(f"Skipping office document (extraction failed): {file_path}")
                    return []
                content = content_result
            elif file_extension == '.html':
                # Use document processor for HTML
                content_result = self.doc_processor._extract_text_from_file(str(file_path))
                if isinstance(content_result, str) and content_result.startswith('{"error"'):
                    if self.verbose:
                        print(f"Skipping HTML file (extraction failed): {file_path}")
                    return []
                content = content_result
            elif file_extension == '.rtf':
                # Use document processor for RTF
                content_result = self.doc_processor._extract_text_from_file(str(file_path))
                if isinstance(content_result, str) and content_result.startswith('{"error"'):
                    if self.verbose:
                        print(f"Skipping RTF file (extraction failed): {file_path}")
                    return []
                content = content_result
            else:
                # Try to read as text file (markdown, txt, code, etc.)
                try:
                    content = file_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    if self.verbose:
                        print(f"Skipping binary file: {file_path}")
                    return []
            
            # Validate content
            if not content or (isinstance(content, str) and len(content.strip()) == 0):
                if self.verbose:
                    print(f"Skipping empty file: {file_path}")
                return []
            
            # Create chunks using document processor - pass content directly, not file path
            chunks = self.doc_processor.create_chunks(
                content=content,  # Pass the actual content, not the file path
                filename=relative_path,
                file_type=file_path.suffix.lstrip('.')
            )
            
            # Add global tags
            if global_tags:
                for chunk in chunks:
                    existing_tags = chunk.get('tags', [])
                    if isinstance(existing_tags, str):
                        existing_tags = [existing_tags]
                    chunk['tags'] = existing_tags + global_tags
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _create_database(self, output_file: str, chunks: List[Dict[str, Any]], 
                        languages: List[str], sources_info: List[str], file_types: List[str]):
        """Create SQLite database with all data"""
        
        # Remove existing file
        if os.path.exists(output_file):
            os.remove(output_file)
        
        conn = sqlite3.connect(output_file)
        cursor = conn.cursor()
        
        try:
            # Create schema
            cursor.execute('''
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    processed_content TEXT NOT NULL,
                    keywords TEXT,
                    language TEXT DEFAULT 'en',
                    embedding BLOB NOT NULL,
                    filename TEXT NOT NULL,
                    section TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    tags TEXT,
                    metadata TEXT,
                    metadata_text TEXT,  -- Searchable text representation of all metadata
                    chunk_hash TEXT UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    processed_content,
                    keywords,
                    metadata_text,
                    content='chunks',
                    content_rowid='id'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE synonyms (
                    word TEXT,
                    pos_tag TEXT,
                    synonyms TEXT,
                    language TEXT DEFAULT 'en',
                    PRIMARY KEY (word, pos_tag, language)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX idx_chunks_filename ON chunks(filename)')
            cursor.execute('CREATE INDEX idx_chunks_language ON chunks(language)')
            cursor.execute('CREATE INDEX idx_chunks_tags ON chunks(tags)')
            
            # Insert config
            embedding_dimensions = 768  # Default for all-mpnet-base-v2
            if chunks and chunks[0].get('embedding'):
                try:
                    if np:
                        embedding_array = np.frombuffer(chunks[0]['embedding'], dtype=np.float32)
                        embedding_dimensions = len(embedding_array)
                except:
                    pass
            
            config_data = {
                'embedding_model': self.model_name,
                'embedding_dimensions': str(embedding_dimensions),
                'chunk_size': str(self.chunk_size),
                'chunk_overlap': str(self.chunk_overlap),
                'preprocessing_version': '1.0',
                'languages': json.dumps(languages),
                'created_at': datetime.now().isoformat(),
                'sources': json.dumps(sources_info),  # Store list of sources instead of single directory
                'file_types': json.dumps(file_types)
            }
            
            for key, value in config_data.items():
                cursor.execute('INSERT INTO config (key, value) VALUES (?, ?)', (key, value))
            
            # Insert chunks
            for chunk in chunks:
                # Create hash for deduplication - include filename, section, and line numbers for uniqueness
                hash_content = f"{chunk['filename']}:{chunk.get('section', '')}:{chunk.get('start_line', 0)}:{chunk.get('end_line', 0)}:{chunk['content']}"
                chunk_hash = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
                
                # Prepare data
                keywords_json = json.dumps(chunk.get('keywords', []))
                tags_json = json.dumps(chunk.get('tags', []))
                
                # Extract metadata from JSON content and merge with chunk metadata
                json_metadata, json_metadata_text = self._extract_metadata_from_json_content(chunk['content'])
                chunk_metadata = chunk.get('metadata', {})
                
                # Merge metadata: chunk metadata takes precedence
                merged_metadata = {**json_metadata, **chunk_metadata}
                metadata_json = json.dumps(merged_metadata)
                
                # Create comprehensive metadata_text including tags
                metadata_text_parts = []
                
                # Add metadata text from JSON content
                if json_metadata_text:
                    metadata_text_parts.append(json_metadata_text)
                
                # Add tags
                tags = chunk.get('tags', [])
                if tags:
                    metadata_text_parts.extend(str(tag).lower() for tag in tags)
                
                # Add section if present
                if chunk.get('section'):
                    metadata_text_parts.append(chunk['section'].lower())
                
                # Add any additional metadata values
                for key, value in chunk_metadata.items():
                    if key not in json_metadata:  # Avoid duplicates
                        metadata_text_parts.append(str(key).lower())
                        if isinstance(value, list):
                            metadata_text_parts.extend(str(v).lower() for v in value)
                        else:
                            metadata_text_parts.append(str(value).lower())
                
                metadata_text = ' '.join(metadata_text_parts)
                
                cursor.execute('''
                    INSERT OR IGNORE INTO chunks (
                        content, processed_content, keywords, language, embedding,
                        filename, section, start_line, end_line, tags, metadata, metadata_text, chunk_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk['content'],
                    chunk.get('processed_content', chunk['content']),
                    keywords_json,
                    chunk.get('language', 'en'),
                    chunk.get('embedding', b''),
                    chunk['filename'],
                    chunk.get('section'),
                    chunk.get('start_line'),
                    chunk.get('end_line'),
                    tags_json,
                    metadata_json,
                    metadata_text,
                    chunk_hash
                ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def validate_index(self, index_file: str) -> Dict[str, Any]:
        """Validate an existing search index"""
        if not os.path.exists(index_file):
            return {"valid": False, "error": "Index file does not exist"}
        
        try:
            conn = sqlite3.connect(index_file)
            cursor = conn.cursor()
            
            # Check schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['chunks', 'chunks_fts', 'synonyms', 'config']
            missing_tables = [t for t in required_tables if t not in tables]
            
            if missing_tables:
                return {"valid": False, "error": f"Missing tables: {missing_tables}"}
            
            # Get config
            cursor.execute("SELECT key, value FROM config")
            config = dict(cursor.fetchall())
            
            # Get chunk count
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Get file count
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
            file_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "valid": True,
                "chunk_count": chunk_count,
                "file_count": file_count,
                "config": config
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _store_chunks_pgvector(self, chunks: List[Dict[str, Any]], collection_name: str,
                              languages: List[str], overwrite: bool = False):
        """
        Store chunks in pgvector backend
        
        Args:
            chunks: List of processed chunks
            collection_name: Name for the collection (from output_file parameter)
            languages: List of supported languages
        """
        from .pgvector_backend import PgVectorBackend
        
        # Extract collection name from the provided name
        if collection_name.endswith('.swsearch'):
            collection_name = collection_name[:-9]  # Remove .swsearch extension
        
        # Clean collection name for PostgreSQL
        import re
        collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', collection_name)
        
        if self.verbose:
            print(f"Storing chunks in pgvector collection: {collection_name}")
        
        # Create backend instance
        backend = PgVectorBackend(self.connection_string)
        
        try:
            # Get embedding dimensions from model
            if self.model:
                embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                embedding_dim = 768  # Default for all-mpnet-base-v2
            
            # Delete existing collection if overwrite is requested
            if overwrite:
                if self.verbose:
                    print(f"Dropping existing collection: {collection_name}")
                backend.delete_collection(collection_name)
            
            # Create schema
            backend.create_schema(collection_name, embedding_dim)
            
            # Convert embeddings from bytes to numpy arrays
            for chunk in chunks:
                if chunk.get('embedding') and isinstance(chunk['embedding'], bytes):
                    if np:
                        chunk['embedding'] = np.frombuffer(chunk['embedding'], dtype=np.float32)
                    else:
                        # If numpy not available, leave as bytes
                        pass
            
            # Prepare config
            config = {
                'model_name': self.model_name,
                'embedding_dimensions': embedding_dim,
                'chunking_strategy': self.chunking_strategy,
                'languages': languages,
                'metadata': {
                    'max_sentences_per_chunk': self.max_sentences_per_chunk,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'index_nlp_backend': self.index_nlp_backend
                }
            }
            
            # Store chunks
            backend.store_chunks(chunks, collection_name, config)
            
            if self.verbose:
                stats = backend.get_stats(collection_name)
                print(f"Stored {stats['total_chunks']} chunks in pgvector")
                print(f"Collection: {collection_name}")
                
        finally:
            backend.close() 