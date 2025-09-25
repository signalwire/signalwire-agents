"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class SearchIndexMigrator:
    """Migrate search indexes between different backends"""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the migrator
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
    
    def migrate_sqlite_to_pgvector(
        self, 
        sqlite_path: str, 
        connection_string: str, 
        collection_name: str,
        overwrite: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Migrate a .swsearch SQLite index to pgvector
        
        Args:
            sqlite_path: Path to .swsearch file
            connection_string: PostgreSQL connection string
            collection_name: Name for the pgvector collection
            overwrite: Whether to overwrite existing collection
            batch_size: Number of chunks to insert at once
            
        Returns:
            Migration statistics
        """
        if not Path(sqlite_path).exists():
            raise FileNotFoundError(f"SQLite index not found: {sqlite_path}")
        
        # Import pgvector backend
        from .pgvector_backend import PgVectorBackend
        
        stats = {
            'source': sqlite_path,
            'target': collection_name,
            'chunks_migrated': 0,
            'errors': 0,
            'config': {}
        }
        
        try:
            # Connect to SQLite
            if self.verbose:
                print(f"Opening SQLite index: {sqlite_path}")
            
            sqlite_conn = sqlite3.connect(sqlite_path)
            cursor = sqlite_conn.cursor()
            
            # Load configuration
            cursor.execute("SELECT key, value FROM config")
            config_rows = cursor.fetchall()
            config = dict(config_rows)
            stats['config'] = config
            
            # Get important config values
            model_name = config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
            embedding_dim = int(config.get('embedding_dimensions', 768))
            
            if self.verbose:
                print(f"Source configuration:")
                print(f"  Model: {model_name}")
                print(f"  Dimensions: {embedding_dim}")
                print(f"  Created: {config.get('created_at', 'Unknown')}")
            
            # Initialize pgvector backend
            pgvector = PgVectorBackend(connection_string)
            
            try:
                # Handle existing collection
                if overwrite:
                    if self.verbose:
                        print(f"Dropping existing collection: {collection_name}")
                    pgvector.delete_collection(collection_name)
                
                # Create schema
                if self.verbose:
                    print(f"Creating pgvector collection: {collection_name}")
                
                pgvector.create_schema(collection_name, embedding_dim)
                
                # Prepare collection config
                collection_config = {
                    'model_name': model_name,
                    'embedding_dimensions': embedding_dim,
                    'chunking_strategy': config.get('chunking_strategy', 'sentence'),
                    'languages': json.loads(config.get('languages', '["en"]')),
                    'metadata': {
                        'migrated_from': sqlite_path,
                        'original_created': config.get('created_at'),
                        'source_dir': config.get('source_dir'),
                        'file_types': json.loads(config.get('file_types', '[]'))
                    }
                }
                
                # Count total chunks
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                
                if self.verbose:
                    print(f"Migrating {total_chunks} chunks...")
                
                # Check if metadata_text column exists (do this once)
                cursor.execute("PRAGMA table_info(chunks)")
                columns = [col[1] for col in cursor.fetchall()]
                has_metadata_text = 'metadata_text' in columns
                
                # Migrate chunks in batches
                offset = 0
                while offset < total_chunks:
                    # Fetch batch of chunks
                    
                    if has_metadata_text:
                        cursor.execute("""
                            SELECT id, content, processed_content, keywords, language, 
                                   embedding, filename, section, start_line, end_line, 
                                   tags, metadata, metadata_text, chunk_hash
                            FROM chunks
                            ORDER BY id
                            LIMIT ? OFFSET ?
                        """, (batch_size, offset))
                    else:
                        cursor.execute("""
                            SELECT id, content, processed_content, keywords, language, 
                                   embedding, filename, section, start_line, end_line, 
                                   tags, metadata, chunk_hash
                            FROM chunks
                            ORDER BY id
                            LIMIT ? OFFSET ?
                        """, (batch_size, offset))
                    
                    chunks_batch = []
                    for row in cursor.fetchall():
                        # Handle both old and new schema (with or without metadata_text)
                        if len(row) == 14:  # New schema with metadata_text
                            (chunk_id, content, processed_content, keywords_json, language,
                             embedding_blob, filename, section, start_line, end_line,
                             tags_json, metadata_json, metadata_text, chunk_hash) = row
                        else:  # Old schema without metadata_text
                            (chunk_id, content, processed_content, keywords_json, language,
                             embedding_blob, filename, section, start_line, end_line,
                             tags_json, metadata_json, chunk_hash) = row
                            metadata_text = None
                        
                        # Convert embedding blob to numpy array if available
                        if embedding_blob and np:
                            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        else:
                            embedding = embedding_blob
                        
                        # Parse JSON fields
                        keywords = json.loads(keywords_json) if keywords_json else []
                        tags = json.loads(tags_json) if tags_json else []
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        
                        chunk = {
                            'content': content,
                            'processed_content': processed_content,
                            'keywords': keywords,
                            'language': language,
                            'embedding': embedding,
                            'filename': filename,
                            'section': section,
                            'start_line': start_line,
                            'end_line': end_line,
                            'tags': tags,
                            'metadata': metadata,
                            'metadata_text': metadata_text,  # Will be regenerated if None
                            'chunk_hash': chunk_hash
                        }
                        
                        chunks_batch.append(chunk)
                    
                    # Store batch in pgvector
                    if chunks_batch:
                        try:
                            pgvector.store_chunks(chunks_batch, collection_name, collection_config)
                            stats['chunks_migrated'] += len(chunks_batch)
                            
                            if self.verbose:
                                progress = (offset + len(chunks_batch)) / total_chunks * 100
                                print(f"  Progress: {stats['chunks_migrated']}/{total_chunks} ({progress:.1f}%)")
                        except Exception as e:
                            logger.error(f"Error storing batch at offset {offset}: {e}")
                            stats['errors'] += len(chunks_batch)
                    
                    offset += batch_size
                
                # Success
                if self.verbose:
                    print(f"\nMigration completed successfully!")
                    print(f"  Chunks migrated: {stats['chunks_migrated']}")
                    print(f"  Errors: {stats['errors']}")
                
            finally:
                pgvector.close()
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            sqlite_conn.close()
        
        return stats
    
    def migrate_pgvector_to_sqlite(
        self,
        connection_string: str,
        collection_name: str,
        output_path: str,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Migrate a pgvector collection to SQLite .swsearch format
        
        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name of the pgvector collection
            output_path: Output .swsearch file path
            batch_size: Number of chunks to fetch at once
            
        Returns:
            Migration statistics
        """
        from .pgvector_backend import PgVectorBackend
        from .index_builder import IndexBuilder
        
        # Ensure output has .swsearch extension
        if not output_path.endswith('.swsearch'):
            output_path += '.swsearch'
        
        stats = {
            'source': f"{collection_name} (pgvector)",
            'target': output_path,
            'chunks_migrated': 0,
            'errors': 0,
            'config': {}
        }
        
        # Connect to pgvector
        if self.verbose:
            print(f"Connecting to pgvector collection: {collection_name}")
        
        pgvector = PgVectorBackend(connection_string)
        
        try:
            # Get collection stats and config
            pg_stats = pgvector.get_stats(collection_name)
            config = pg_stats.get('config', {})
            stats['config'] = config
            
            total_chunks = pg_stats.get('total_chunks', 0)
            
            if self.verbose:
                print(f"Source configuration:")
                print(f"  Model: {config.get('model_name', 'Unknown')}")
                print(f"  Dimensions: {config.get('embedding_dimensions', 'Unknown')}")
                print(f"  Total chunks: {total_chunks}")
            
            # Create SQLite database structure
            # We'll manually create it to match the expected format
            if Path(output_path).exists():
                Path(output_path).unlink()
            
            conn = sqlite3.connect(output_path)
            cursor = conn.cursor()
            
            # Create schema (matching index_builder.py)
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
                    chunk_hash TEXT UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    processed_content,
                    keywords,
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
            
            # Create indexes
            cursor.execute('CREATE INDEX idx_chunks_filename ON chunks(filename)')
            cursor.execute('CREATE INDEX idx_chunks_language ON chunks(language)')
            cursor.execute('CREATE INDEX idx_chunks_tags ON chunks(tags)')
            
            # Insert config
            config_data = {
                'embedding_model': config.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
                'embedding_dimensions': str(config.get('embedding_dimensions', 768)),
                'chunk_size': str(config.get('metadata', {}).get('chunk_size', 50)),
                'chunk_overlap': str(config.get('metadata', {}).get('chunk_overlap', 10)),
                'preprocessing_version': '1.0',
                'languages': json.dumps(config.get('languages', ['en'])),
                'created_at': datetime.now().isoformat(),
                'source_dir': config.get('metadata', {}).get('source_dir', 'pgvector_migration'),
                'file_types': json.dumps(config.get('metadata', {}).get('file_types', []))
            }
            
            for key, value in config_data.items():
                cursor.execute('INSERT INTO config (key, value) VALUES (?, ?)', (key, value))
            
            # TODO: Implement chunk fetching from pgvector
            # This would require adding a method to PgVectorBackend to fetch chunks
            # For now, we'll note this as a limitation
            
            if self.verbose:
                print("\nNote: pgvector to SQLite migration requires implementing chunk fetching in PgVectorBackend")
                print("This feature is planned for future development.")
            
            conn.commit()
            conn.close()
            
        finally:
            pgvector.close()
        
        return stats
    
    def get_index_info(self, index_path: str) -> Dict[str, Any]:
        """
        Get information about a search index
        
        Args:
            index_path: Path to index file or pgvector collection identifier
            
        Returns:
            Index information including type, config, and statistics
        """
        info = {}
        
        if index_path.endswith('.swsearch') and Path(index_path).exists():
            # SQLite index
            info['type'] = 'sqlite'
            info['path'] = index_path
            
            conn = sqlite3.connect(index_path)
            cursor = conn.cursor()
            
            # Get config
            cursor.execute("SELECT key, value FROM config")
            info['config'] = dict(cursor.fetchall())
            
            # Get stats
            cursor.execute("SELECT COUNT(*) FROM chunks")
            info['total_chunks'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
            info['total_files'] = cursor.fetchone()[0]
            
            conn.close()
            
        else:
            info['type'] = 'unknown'
            info['path'] = index_path
        
        return info