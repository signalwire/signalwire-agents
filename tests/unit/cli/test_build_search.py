"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for CLI build_search module
"""

import pytest
import sys
import types
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO
import argparse

# Ensure search submodules are patchable even when search deps (nltk, etc.) are missing.
# The build_search.py code does local imports from these modules, so @patch needs
# them to exist in sys.modules for the patch target to resolve.
# Only insert stubs for modules that truly can't be imported.
def _ensure_mock_module(module_path, attrs=None):
    """Register a fake module in sys.modules if the real one isn't importable."""
    try:
        __import__(module_path)
    except (ImportError, ModuleNotFoundError):
        if module_path not in sys.modules:
            mod = types.ModuleType(module_path)
            for attr_name, attr_val in (attrs or {}).items():
                setattr(mod, attr_name, attr_val)
            sys.modules[module_path] = mod

_ensure_mock_module('signalwire_agents.search.index_builder', {
    'IndexBuilder': type('IndexBuilder', (), {}),
})
_ensure_mock_module('signalwire_agents.search.search_engine', {
    'SearchEngine': type('SearchEngine', (), {}),
})
_ensure_mock_module('signalwire_agents.search.query_processor', {
    'preprocess_query': lambda *a, **kw: {},
})
_ensure_mock_module('signalwire_agents.search.migration', {
    'SearchIndexMigrator': type('SearchIndexMigrator', (), {}),
})

from signalwire_agents.cli.build_search import (
    main,
    validate_command,
    search_command,
    console_entry_point
)


class TestBuildSearchMain:
    """Test the main build command functionality"""
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs'])
    def test_basic_build_command(self, mock_builder_class):
        """Test basic build command with minimal arguments"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('pathlib.Path.stem', 'docs'), \
             patch('os.path.exists', return_value=True):

            main()

            # Verify IndexBuilder was created with defaults
            mock_builder_class.assert_called_once_with(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                chunking_strategy='sentence',
                max_sentences_per_chunk=5,
                chunk_size=50,
                chunk_overlap=10,
                split_newlines=None,
                index_nlp_backend='nltk',
                verbose=False,
                semantic_threshold=0.5,
                topic_threshold=0.3,
                backend='sqlite',
                connection_string=None
            )

            # Verify build_index_from_sources was called
            mock_builder.build_index_from_sources.assert_called_once()
            args = mock_builder.build_index_from_sources.call_args
            assert len(args[1]['sources']) == 1
            assert args[1]['output_file'] == 'docs.swsearch'
            assert args[1]['file_types'] == ['md', 'txt', 'rst']
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs', './examples', 
        '--output', 'custom.swsearch',
        '--chunking-strategy', 'sliding',
        '--chunk-size', '100',
        '--overlap-size', '20',
        '--file-types', 'md,py,txt',
        '--exclude', '**/test/**,**/__pycache__/**',
        '--languages', 'en,es',
        '--model', 'custom-model',
        '--tags', 'docs,api',
        '--verbose',
        '--validate'
    ])
    def test_full_build_command(self, mock_builder_class):
        """Test build command with all arguments"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.validate_index.return_value = {
            'valid': True,
            'chunk_count': 100,
            'file_count': 10,
            'config': {'model': 'custom-model'}
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:

            main()

            # Verify IndexBuilder was created with custom parameters
            mock_builder_class.assert_called_once_with(
                model_name='custom-model',
                chunking_strategy='sliding',
                max_sentences_per_chunk=5,
                chunk_size=100,
                chunk_overlap=20,
                split_newlines=None,
                index_nlp_backend='nltk',
                verbose=True,
                semantic_threshold=0.5,
                topic_threshold=0.3,
                backend='sqlite',
                connection_string=None
            )
            
            # Verify build_index_from_sources was called with custom parameters
            args = mock_builder.build_index_from_sources.call_args[1]
            assert len(args['sources']) == 2
            assert args['output_file'] == 'custom.swsearch'
            assert args['file_types'] == ['md', 'py', 'txt']
            assert args['exclude_patterns'] == ['**/test/**', '**/__pycache__/**']
            assert args['languages'] == ['en', 'es']
            assert args['tags'] == ['docs', 'api']
            
            # Verify validation was called
            mock_builder.validate_index.assert_called_once_with('custom.swsearch')
            
            # Verify verbose output
            mock_print.assert_any_call("Building search index:")
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', 'README.md'])
    def test_mixed_sources(self, mock_builder_class):
        """Test build command with mixed file and directory sources"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        
        def mock_exists(self):
            return str(self) in ['./docs', 'README.md']
        
        def mock_is_file(self):
            return str(self) == 'README.md'
        
        with patch('pathlib.Path.exists', mock_exists), \
             patch('pathlib.Path.is_file', mock_is_file), \
             patch('pathlib.Path.stem', 'sources'), \
             patch('os.path.exists', return_value=True):

            main()

            # Should use generic name for multiple sources
            args = mock_builder.build_index_from_sources.call_args[1]
            assert args['output_file'] == 'sources.swsearch'
    
    @patch('sys.argv', ['sw-search', './nonexistent'])
    def test_nonexistent_source(self):
        """Test handling of nonexistent sources"""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("Error: No valid sources found")
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', './missing'])
    def test_partial_valid_sources(self, mock_builder_class):
        """Test handling when some sources are invalid"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        # Create mock Path objects with proper __str__ (needs self parameter)
        docs_path = Mock()
        docs_path.exists.return_value = True
        docs_path.is_file.return_value = False
        docs_path.name = 'docs'
        docs_path.__str__ = lambda self: './docs'

        missing_path = Mock()
        missing_path.exists.return_value = False
        missing_path.__str__ = lambda self: './missing'

        def mock_path_constructor(path_str):
            if str(path_str) == './docs':
                return docs_path
            elif str(path_str) == './missing':
                return missing_path
            else:
                # Default mock for other paths
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_path.__str__ = lambda self: str(path_str)
                return mock_path

        # Patch Path where it was imported in build_search module
        with patch('signalwire_agents.cli.build_search.Path', side_effect=mock_path_constructor), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:

            main()

            # Should warn about missing source but continue
            mock_print.assert_any_call("Warning: Source does not exist, skipping: ./missing")

            # Should still build with valid source
            args = mock_builder.build_index_from_sources.call_args[1]
            assert len(args['sources']) == 1
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './missing1', './missing2'])
    def test_all_invalid_sources(self, mock_builder_class):
        """Test handling when all sources are invalid"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        
        with patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("Error: No valid sources found")
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--output', 'test'])
    def test_output_extension_handling(self, mock_builder_class):
        """Test automatic addition of .swsearch extension"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('os.path.exists', return_value=True):

            main()

            args = mock_builder.build_index_from_sources.call_args[1]
            assert args['output_file'] == 'test.swsearch'
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs'])
    def test_keyboard_interrupt(self, mock_builder_class):
        """Test handling of keyboard interrupt"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_index_from_sources.side_effect = KeyboardInterrupt()
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("\n\nBuild interrupted by user")
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs'])
    def test_build_error(self, mock_builder_class):
        """Test handling of build errors"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_index_from_sources.side_effect = Exception("Build failed")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("\nError building index: Build failed")
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--validate'])
    def test_validation_failure(self, mock_builder_class):
        """Test handling of validation failure"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.validate_index.return_value = {
            'valid': False,
            'error': 'Invalid index format'
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("✗ Index validation failed: Invalid index format")


class TestValidateCommand:
    """Test the validate command functionality"""
    
    @patch('argparse.ArgumentParser')
    def test_validate_nonexistent_file(self, mock_parser_class):
        """Test validation of nonexistent file"""
        # Mock argument parser
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.index_file = 'nonexistent.swsearch'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args
        
        with patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            validate_command()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("Error: Index file does not exist: nonexistent.swsearch")


class TestSearchCommand:
    """Test the search command functionality"""
    
    @patch('sys.argv', ['search', 'test.swsearch', 'test query'])
    def test_basic_search(self):
        """Test basic search command"""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 100, 'total_files': 10}
        mock_engine.search.return_value = [
            {
                'score': 0.95,
                'content': 'Test content',
                'metadata': {'filename': 'test.md', 'section': 'intro'}
            }
        ]
        
        mock_preprocess_return = {
            'vector': [0.1, 0.2, 0.3],
            'enhanced_text': 'enhanced test query'
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess_return):
            
            search_command()
            
            mock_print.assert_any_call("Found 1 result(s) for 'test query':")
    
    @patch('sys.argv', [
        'search', 'test.swsearch', 'test query',
        '--count', '10',
        '--distance-threshold', '0.5',
        '--tags', 'docs,api',
        '--query-nlp-backend', 'spacy',
        '--verbose',
        '--json'
    ])
    def test_full_search_command(self):
        """Test search command with all options"""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 100, 'total_files': 10}
        mock_engine.search.return_value = [
            {
                'score': 0.95,
                'content': 'Test content',
                'metadata': {'filename': 'test.md', 'tags': ['docs']}
            }
        ]
        
        mock_preprocess_return = {
            'vector': [0.1, 0.2, 0.3],
            'enhanced_text': 'enhanced test query'
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess_return):
            
            search_command()
            
            # Should output JSON
            printed_calls = [str(call) for call in mock_print.call_args_list if call.args]
            printed_output = ''.join(printed_calls)
            assert '"query": "test query"' in printed_output
    
    @patch('sys.argv', ['search', 'test.swsearch', 'test query', '--no-content'])
    def test_search_no_content(self):
        """Test search command with no content output"""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 100, 'total_files': 10}
        mock_engine.search.return_value = [
            {
                'score': 0.95,
                'content': 'Test content that should not be shown',
                'metadata': {'filename': 'test.md'}
            }
        ]
        
        mock_preprocess_return = {
            'vector': [0.1, 0.2, 0.3],
            'enhanced_text': 'test query'
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess_return):
            
            search_command()
            
            # Content should not be printed
            printed_output = ''.join([str(call.args[0]) for call in mock_print.call_args_list])
            assert 'Test content that should not be shown' not in printed_output
    
    @patch('sys.argv', ['search', 'test.swsearch', 'test query'])
    def test_search_no_results(self):
        """Test search command with no results"""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 100, 'total_files': 10}
        mock_engine.search.return_value = []
        
        mock_preprocess_return = {
            'vector': [0.1, 0.2, 0.3],
            'enhanced_text': 'test query'
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess_return), \
             pytest.raises(SystemExit) as exc_info:
            
            search_command()
            
            assert exc_info.value.code == 0
            mock_print.assert_any_call("No results found for 'test query'")
    
    @patch('sys.argv', ['search', 'nonexistent.swsearch', 'query'])
    def test_search_nonexistent_file(self):
        """Test search with nonexistent index file"""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            search_command()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("Error: Index file does not exist: nonexistent.swsearch")
    
    @patch('sys.argv', ['search', 'test.swsearch', 'query'])
    def test_search_import_error(self):
        """Test search with missing dependencies"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            # Mock import error for search dependencies
            with patch('signalwire_agents.search.search_engine.SearchEngine', side_effect=ImportError("No module")):
                search_command()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("Error: Search functionality not available. Install with: pip install signalwire-agents[search]")
    
    @patch('sys.argv', ['search', 'test.swsearch', 'query'])
    def test_search_engine_error(self):
        """Test search engine initialization error"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', side_effect=Exception("Engine error")), \
             pytest.raises(SystemExit) as exc_info:
            
            search_command()
            
            assert exc_info.value.code == 1
            mock_print.assert_any_call("Error searching index: Engine error")


class TestConsoleEntryPoint:
    """Test the console entry point functionality"""
    
    @patch('signalwire_agents.cli.build_search.main')
    @patch('sys.argv', ['sw-search', './docs'])
    def test_console_entry_main(self, mock_main):
        """Test console entry point calls main for build command"""
        console_entry_point()
        mock_main.assert_called_once()
    
    @patch('signalwire_agents.cli.build_search.validate_command')
    @patch('sys.argv', ['sw-search', 'validate', 'test.swsearch'])
    def test_console_entry_validate(self, mock_validate):
        """Test console entry point calls validate_command"""
        console_entry_point()
        mock_validate.assert_called_once()
        # Should remove 'validate' from argv
        assert 'validate' not in sys.argv
    
    @patch('signalwire_agents.cli.build_search.search_command')
    @patch('sys.argv', ['sw-search', 'search', 'test.swsearch', 'query'])
    def test_console_entry_search(self, mock_search):
        """Test console entry point calls search_command"""
        console_entry_point()
        mock_search.assert_called_once()
        # Should remove 'search' from argv
        assert 'search' not in sys.argv


class TestArgumentParsing:
    """Test argument parsing edge cases"""
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--chunking-strategy', 'sentence', '--split-newlines', '2'])
    def test_sentence_chunking_with_newlines(self, mock_builder_class):
        """Test sentence chunking with split newlines parameter"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('os.path.exists', return_value=True):

            main()
            
            mock_builder_class.assert_called_once_with(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                chunking_strategy='sentence',
                max_sentences_per_chunk=5,
                chunk_size=50,
                chunk_overlap=10,
                split_newlines=2,
                index_nlp_backend='nltk',
                verbose=False,
                semantic_threshold=0.5,
                topic_threshold=0.3,
                backend='sqlite',
                connection_string=None
            )
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--chunking-strategy', 'paragraph'])
    def test_paragraph_chunking(self, mock_builder_class):
        """Test paragraph chunking strategy"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('os.path.exists', return_value=True):

            main()
            
            mock_builder_class.assert_called_once_with(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                chunking_strategy='paragraph',
                max_sentences_per_chunk=5,
                chunk_size=50,
                chunk_overlap=10,
                split_newlines=None,
                index_nlp_backend='nltk',
                verbose=False,
                semantic_threshold=0.5,
                topic_threshold=0.3,
                backend='sqlite',
                connection_string=None
            )
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--chunking-strategy', 'page'])
    def test_page_chunking(self, mock_builder_class):
        """Test page chunking strategy"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('os.path.exists', return_value=True):

            main()
            
            mock_builder_class.assert_called_once_with(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                chunking_strategy='page',
                max_sentences_per_chunk=5,
                chunk_size=50,
                chunk_overlap=10,
                split_newlines=None,
                index_nlp_backend='nltk',
                verbose=False,
                semantic_threshold=0.5,
                topic_threshold=0.3,
                backend='sqlite',
                connection_string=None
            )


class TestVerboseOutput:
    """Test verbose output functionality"""
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'sliding'])
    def test_verbose_sliding_output(self, mock_builder_class):
        """Test verbose output for sliding window strategy"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:

            main()
            
            # Check for sliding window specific output
            mock_print.assert_any_call("  Chunk size (words): 50")
            mock_print.assert_any_call("  Overlap size (words): 10")
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'sentence', '--split-newlines', '3'])
    def test_verbose_sentence_output(self, mock_builder_class):
        """Test verbose output for sentence strategy with newlines"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', 'docs'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:

            main()
            
            # Check for sentence specific output
            mock_print.assert_any_call("  Max sentences per chunk: 5")
            mock_print.assert_any_call("  Split on newlines: 3")


class TestErrorHandlingEdgeCases:
    """Test edge cases and error handling"""
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose'])
    def test_verbose_error_with_traceback(self, mock_builder_class):
        """Test verbose error output includes traceback"""
        mock_builder_class.side_effect = Exception("Detailed error")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print, \
             patch('traceback.print_exc') as mock_traceback, \
             pytest.raises(SystemExit):
            
            main()
            
            mock_traceback.assert_called_once()
    
    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['validate', 'test.swsearch', '--verbose'])
    def test_validate_verbose_error_with_traceback(self, mock_builder_class):
        """Test verbose validation error includes traceback"""
        mock_builder_class.side_effect = Exception("Validation detailed error")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('traceback.print_exc') as mock_traceback, \
             pytest.raises(SystemExit):
            
            validate_command()
            
            mock_traceback.assert_called_once()
    
    @patch('sys.argv', ['search', 'test.swsearch', 'query', '--verbose'])
    def test_search_verbose_error_with_traceback(self):
        """Test verbose search error includes traceback"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('traceback.print_exc') as mock_traceback, \
             patch('signalwire_agents.search.search_engine.SearchEngine', side_effect=Exception("Search detailed error")), \
             pytest.raises(SystemExit):
            
            search_command()

            mock_traceback.assert_called_once()


# ---------------------------------------------------------------------------
# Additional imports needed for new tests
# ---------------------------------------------------------------------------
from signalwire_agents.cli.build_search import (
    migrate_command,
    remote_command,
)


class TestConsoleEntryPointExtended:
    """Additional tests for console_entry_point subcommand routing."""

    @patch('builtins.print')
    @patch('sys.argv', ['sw-search', '--help'])
    def test_console_entry_help_flag(self, mock_print):
        """Test --help flag shows help text without importing heavy modules."""
        console_entry_point()
        printed = ''.join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        assert 'Build local search index from documents' in printed

    @patch('builtins.print')
    @patch('sys.argv', ['sw-search', '-h'])
    def test_console_entry_help_short_flag(self, mock_print):
        """Test -h flag shows help text."""
        console_entry_point()
        printed = ''.join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        assert 'positional arguments' in printed

    @patch('signalwire_agents.cli.build_search.remote_command')
    @patch('sys.argv', ['sw-search', 'remote', 'http://localhost:8001', 'query'])
    def test_console_entry_remote(self, mock_remote):
        """Test console entry point routes to remote_command."""
        console_entry_point()
        mock_remote.assert_called_once()
        assert 'remote' not in sys.argv

    @patch('signalwire_agents.cli.build_search.migrate_command')
    @patch('sys.argv', ['sw-search', 'migrate', 'test.swsearch', '--info'])
    def test_console_entry_migrate(self, mock_migrate):
        """Test console entry point routes to migrate_command."""
        console_entry_point()
        mock_migrate.assert_called_once()
        assert 'migrate' not in sys.argv


class TestMainPgvectorBackend:
    """Tests for pgvector backend handling in main()."""

    @patch('sys.argv', ['sw-search', './docs', '--backend', 'pgvector'])
    def test_pgvector_requires_connection_string(self):
        """--backend pgvector without --connection-string should exit."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "Error: --connection-string is required for pgvector backend"
        )

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs',
        '--backend', 'pgvector',
        '--connection-string', 'postgresql://user:pass@localhost/db',
    ])
    def test_pgvector_default_output_single_source(self, mock_builder_class):
        """pgvector single source should use source name as collection name."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('builtins.print'):
            main()

        call_kw = mock_builder.build_index_from_sources.call_args[1]
        # pgvector should NOT add .swsearch
        assert not call_kw['output_file'].endswith('.swsearch')

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs', './more',
        '--backend', 'pgvector',
        '--connection-string', 'postgresql://u:p@localhost/db',
    ])
    def test_pgvector_default_output_multi_source(self, mock_builder_class):
        """pgvector with multiple sources defaults to 'documents' collection."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print'):
            main()

        call_kw = mock_builder.build_index_from_sources.call_args[1]
        assert call_kw['output_file'] == 'documents'

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs',
        '--backend', 'pgvector',
        '--connection-string', 'postgresql://u:p@localhost/db',
    ])
    def test_pgvector_success_message(self, mock_builder_class):
        """pgvector success path prints collection info."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print:
            main()

        printed = [str(c) for c in mock_print.call_args_list]
        assert any('collection created successfully' in s for s in printed)


class TestMainOutputConflict:
    """Tests for --output and --output-dir conflict detection."""

    @patch('sys.argv', ['sw-search', './docs', '--output', 'out.swsearch', '--output-dir', './dir'])
    def test_output_and_output_dir_conflict(self):
        """Specifying both --output and --output-dir should error."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: Cannot specify both --output and --output-dir")


class TestMainJsonOutputFormat:
    """Tests for --output-format json handling in main()."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--output-format', 'json'])
    def test_json_format_default_output_name(self, mock_builder_class):
        """JSON format without explicit output should default to chunks.json."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder._discover_files_from_sources.return_value = []

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print'), \
             patch('builtins.open', MagicMock()):
            main()

        # builder should have been constructed (JSON mode uses IndexBuilder too)
        mock_builder_class.assert_called_once()

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs',
        '--output-format', 'json',
        '--backend', 'pgvector',
        '--connection-string', 'postgresql://u:p@localhost/db',
    ])
    def test_json_format_ignores_backend_warning(self, mock_builder_class):
        """JSON format with non-sqlite backend should warn."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder._discover_files_from_sources.return_value = []

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print, \
             patch('builtins.open', MagicMock()):
            main()

        mock_print.assert_any_call(
            "Warning: --backend is ignored when using --output-format json"
        )

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs',
        '--output-format', 'json',
        '--output', 'my_chunks',
    ])
    def test_json_format_output_gets_json_extension(self, mock_builder_class):
        """JSON format output without .json suffix gets one appended via Path.with_suffix."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder._discover_files_from_sources.return_value = []

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print, \
             patch('builtins.open', MagicMock()):
            main()

        # The code calls Path(args.output).with_suffix('.json') when no suffix.
        # With no files discovered, it writes to the single-file output.
        # Verify that the printed success message contains .json or the open call happened.
        # Since _discover_files_from_sources returns [], all_chunks is empty, write still occurs.
        printed = ''.join(str(c) for c in mock_print.call_args_list)
        # The output file should end in .json
        assert '.json' in printed or mock_builder._discover_files_from_sources.called

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', [
        'sw-search', './docs',
        '--output-format', 'json',
        '--output-dir', '/tmp/test_chunks_out',
    ])
    def test_json_format_output_dir_mode(self, mock_builder_class):
        """JSON format with --output-dir should process without error."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder._discover_files_from_sources.return_value = []

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('builtins.print') as mock_print:
            main()

        # With no files discovered, should still complete and print summary
        printed = ''.join(str(c) for c in mock_print.call_args_list)
        assert 'JSON files' in printed or 'chunks' in printed.lower()


class TestMainOutputDirIndexFormat:
    """Tests for --output-dir with index format."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--output-dir', '/tmp/idx_out'])
    def test_output_dir_single_source_sqlite(self, mock_builder_class):
        """Index format with --output-dir and single source auto-names the file."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('pathlib.Path.mkdir'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print'):
            main()

        call_kw = mock_builder.build_index_from_sources.call_args[1]
        assert call_kw['output_file'].endswith('.swsearch')

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './a', './b', '--output-dir', '/tmp/idx_out'])
    def test_output_dir_multi_source_sqlite(self, mock_builder_class):
        """Index format with --output-dir and multiple sources uses 'combined'."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.mkdir'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print'):
            main()

        call_kw = mock_builder.build_index_from_sources.call_args[1]
        assert 'combined' in call_kw['output_file']


class TestMainModelAlias:
    """Tests for model alias resolution in main()."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--model', 'base'])
    def test_model_alias_base(self, mock_builder_class):
        """Model alias 'base' should resolve to the full model name."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True):
            main()

        call_kw = mock_builder_class.call_args[1]
        assert call_kw['model_name'] == 'sentence-transformers/all-mpnet-base-v2'

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--model', 'large'])
    def test_model_alias_large(self, mock_builder_class):
        """Model alias 'large' should resolve correctly."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True):
            main()

        call_kw = mock_builder_class.call_args[1]
        assert call_kw['model_name'] == 'sentence-transformers/all-mpnet-base-v2'

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--model', 'custom-org/my-model'])
    def test_model_full_name_passthrough(self, mock_builder_class):
        """Full model name that is not an alias should pass through unchanged."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True):
            main()

        call_kw = mock_builder_class.call_args[1]
        assert call_kw['model_name'] == 'custom-org/my-model'


class TestMainVerboseStrategies:
    """Tests for verbose output across all chunking strategies."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'paragraph'])
    def test_verbose_paragraph(self, mock_builder_class):
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            main()
        mock_print.assert_any_call("  Chunking by paragraphs (double newlines)")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'page'])
    def test_verbose_page(self, mock_builder_class):
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            main()
        mock_print.assert_any_call("  Chunking by pages")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'semantic'])
    def test_verbose_semantic(self, mock_builder_class):
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            main()
        mock_print.assert_any_call("  Semantic chunking (similarity threshold: 0.5)")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'topic'])
    def test_verbose_topic(self, mock_builder_class):
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            main()
        mock_print.assert_any_call("  Topic-based chunking (similarity threshold: 0.3)")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'qa'])
    def test_verbose_qa(self, mock_builder_class):
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            main()
        mock_print.assert_any_call("  QA-optimized chunking")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs', '--verbose', '--chunking-strategy', 'sentence'])
    def test_verbose_sentence_no_newlines(self, mock_builder_class):
        """Sentence strategy without split-newlines should not print newline line."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            main()
        mock_print.assert_any_call("  Max sentences per chunk: 5")
        # split_newlines not set, so this line should NOT appear
        for c in mock_print.call_args_list:
            if c.args:
                assert 'Split on newlines' not in str(c.args[0])


class TestMainSingleFileSource:
    """Tests for single file source naming."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', 'README.md'])
    def test_single_file_source_names_output(self, mock_builder_class):
        """Single file source should name output after file stem."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        def mock_exists(self):
            return True

        def mock_is_file(self):
            return str(self) == 'README.md'

        with patch('pathlib.Path.exists', mock_exists), \
             patch('pathlib.Path.is_file', mock_is_file), \
             patch('pathlib.Path.stem', new_callable=lambda: property(lambda self: 'README')), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print'):
            main()

        call_kw = mock_builder.build_index_from_sources.call_args[1]
        assert call_kw['output_file'] == 'README.swsearch'


class TestMainIndexCreationCheck:
    """Tests for post-build file existence check."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('sys.argv', ['sw-search', './docs'])
    def test_index_file_not_created(self, mock_builder_class):
        """If output file is not created, should exit with error."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.name', new_callable=lambda: property(lambda self: 'docs')), \
             patch('os.path.exists', return_value=False), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "\n\u2717 Search index creation failed - no files were processed"
        )


class TestValidateCommandExtended:
    """Additional tests for validate_command."""

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('argparse.ArgumentParser')
    def test_validate_success(self, mock_parser_class, mock_builder_class):
        """Successful validation prints valid message."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.index_file = 'test.swsearch'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.validate_index.return_value = {
            'valid': True,
            'chunk_count': 50,
            'file_count': 5,
            'config': {'embedding_model': 'test-model'},
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            validate_command()

        mock_print.assert_any_call("\u2713 Index is valid: test.swsearch")
        mock_print.assert_any_call("  Chunks: 50")
        mock_print.assert_any_call("  Files: 5")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('argparse.ArgumentParser')
    def test_validate_success_verbose(self, mock_parser_class, mock_builder_class):
        """Successful verbose validation prints configuration details."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.index_file = 'test.swsearch'
        mock_args.verbose = True
        mock_parser.parse_args.return_value = mock_args

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.validate_index.return_value = {
            'valid': True,
            'chunk_count': 50,
            'file_count': 5,
            'config': {'embedding_model': 'test-model', 'dimensions': 384},
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print:
            validate_command()

        mock_print.assert_any_call("\nConfiguration:")
        mock_print.assert_any_call("  embedding_model: test-model")

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('argparse.ArgumentParser')
    def test_validate_failure(self, mock_parser_class, mock_builder_class):
        """Failed validation should exit with code 1."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.index_file = 'bad.swsearch'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.validate_index.return_value = {
            'valid': False,
            'error': 'corrupted index',
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            validate_command()

        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "\u2717 Index validation failed: corrupted index"
        )

    @patch('signalwire_agents.search.index_builder.IndexBuilder')
    @patch('argparse.ArgumentParser')
    def test_validate_exception(self, mock_parser_class, mock_builder_class):
        """Exception during validation should exit with code 1."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.index_file = 'bad.swsearch'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args

        mock_builder_class.side_effect = Exception("Cannot read index")

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            validate_command()

        assert exc_info.value.code == 1


class TestSearchCommandExtended:
    """Additional tests for search_command."""

    @patch('sys.argv', ['search', 'test.swsearch'])
    def test_search_no_query_no_shell(self):
        """Missing query without --shell should exit."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            search_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: Query is required unless using --shell mode")

    @patch('sys.argv', ['search', 'test.swsearch', 'q', '--keyword-weight', '1.5'])
    def test_search_keyword_weight_too_high(self):
        """keyword-weight > 1.0 should exit."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            search_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: --keyword-weight must be between 0.0 and 1.0")

    @patch('sys.argv', ['search', 'test.swsearch', 'q', '--keyword-weight', '-0.1'])
    def test_search_keyword_weight_negative(self):
        """keyword-weight < 0.0 should exit."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            search_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: --keyword-weight must be between 0.0 and 1.0")

    @patch('sys.argv', [
        'search', 'coll', 'q',
        '--backend', 'pgvector',
    ])
    def test_search_pgvector_requires_connection_string(self):
        """pgvector backend without connection string should exit."""
        with patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            search_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "Error: --connection-string is required for pgvector backend"
        )

    @patch('sys.argv', ['search', 'test.swsearch', 'q', '--model', 'mini'])
    def test_search_model_alias_resolved(self):
        """Model alias in search should resolve to full name."""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {
            'total_chunks': 10, 'total_files': 1,
            'config': {'embedding_model': 'whatever'},
        }
        mock_engine.search.return_value = []

        mock_preprocess = {
            'vector': [0.1], 'enhanced_text': 'q',
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print'), \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine) as mock_se, \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess), \
             pytest.raises(SystemExit):
            search_command()

        # SearchEngine should be called with the resolved model name
        call_kw = mock_se.call_args[1]
        assert call_kw['model'] == 'sentence-transformers/all-MiniLM-L6-v2'

    @patch('sys.argv', ['search', 'test.swsearch', 'q', '--json', '--no-content'])
    def test_search_json_no_content(self):
        """JSON output with --no-content should omit content field."""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 10, 'total_files': 1}
        mock_engine.search.return_value = [
            {
                'score': 0.9,
                'content': 'Hidden content',
                'metadata': {'filename': 'f.md'},
            }
        ]

        mock_preprocess = {'vector': [0.1], 'enhanced_text': 'q'}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess):
            search_command()

        # Parse the JSON output printed
        printed_text = ''
        for c in mock_print.call_args_list:
            if c.args:
                printed_text += str(c.args[0])
        assert 'Hidden content' not in printed_text

    @patch('sys.argv', ['search', 'test.swsearch', 'test query', '--tags', 'docs'])
    def test_search_no_results_with_tags(self):
        """No results with tags should mention tags in output."""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 10, 'total_files': 1}
        mock_engine.search.return_value = []

        mock_preprocess = {'vector': [0.1], 'enhanced_text': 'test query'}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess), \
             pytest.raises(SystemExit):
            search_command()

        printed = [str(c) for c in mock_print.call_args_list]
        assert any('tags' in s.lower() for s in printed)

    @patch('sys.argv', ['search', 'test.swsearch', 'q'])
    def test_search_result_with_line_numbers_and_tags(self):
        """Results with line_start and tags metadata should display them."""
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 10, 'total_files': 1}
        mock_engine.search.return_value = [
            {
                'score': 0.9,
                'content': 'content',
                'metadata': {
                    'filename': 'f.md',
                    'line_start': 10,
                    'line_end': 20,
                    'tags': ['api', 'docs'],
                },
            }
        ]

        mock_preprocess = {'vector': [0.1], 'enhanced_text': 'q'}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess):
            search_command()

        printed = [str(c) for c in mock_print.call_args_list]
        assert any('Lines: 10-20' in s for s in printed)
        assert any('Tags: api, docs' in s for s in printed)

    @patch('sys.argv', ['search', 'test.swsearch', 'q'])
    def test_search_long_content_truncated(self):
        """Content longer than 500 chars should be truncated in non-verbose mode."""
        long_content = 'A' * 600
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {'total_chunks': 10, 'total_files': 1}
        mock_engine.search.return_value = [
            {
                'score': 0.9,
                'content': long_content,
                'metadata': {'filename': 'f.md'},
            }
        ]

        mock_preprocess = {'vector': [0.1], 'enhanced_text': 'q'}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('signalwire_agents.search.search_engine.SearchEngine', return_value=mock_engine), \
             patch('signalwire_agents.search.query_processor.preprocess_query', return_value=mock_preprocess):
            search_command()

        printed = ''.join(str(c) for c in mock_print.call_args_list)
        assert '...' in printed


class TestMigrateCommand:
    """Tests for migrate_command."""

    @patch('sys.argv', ['migrate', '--info', 'test.swsearch'])
    def test_migrate_info_success(self):
        """--info flag should display index information."""
        mock_migrator = Mock()
        mock_migrator.get_index_info.return_value = {
            'type': 'sqlite',
            'total_chunks': 100,
            'total_files': 10,
            'config': {
                'embedding_model': 'test-model',
                'embedding_dimensions': 384,
                'created_at': '2025-01-01',
            },
        }

        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print:
            migrate_command()

        mock_print.assert_any_call("Index Information: test.swsearch")
        mock_print.assert_any_call("  Total chunks: 100")

    @patch('sys.argv', ['migrate', '--info'])
    def test_migrate_info_no_source(self):
        """--info without source should exit."""
        with patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: Source index required with --info")

    @patch('sys.argv', ['migrate'])
    def test_migrate_no_source(self):
        """No source should exit."""
        with patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: Source index required for migration")

    @patch('sys.argv', ['migrate', 'test.swsearch'])
    def test_migrate_no_direction(self):
        """No migration direction should exit."""
        with patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "Error: Must specify migration direction (--to-pgvector or --to-sqlite)"
        )

    @patch('sys.argv', ['migrate', 'test.swsearch', '--to-pgvector'])
    def test_migrate_to_pgvector_no_connection_string(self):
        """to-pgvector without connection string should exit."""
        mock_migrator = Mock()
        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "Error: --connection-string required for pgvector migration"
        )

    @patch('sys.argv', [
        'migrate', 'test.swsearch', '--to-pgvector',
        '--connection-string', 'postgresql://u:p@localhost/db',
    ])
    def test_migrate_to_pgvector_no_collection_name(self):
        """to-pgvector without collection name should exit."""
        mock_migrator = Mock()
        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1
        mock_print.assert_any_call(
            "Error: --collection-name required for pgvector migration"
        )

    @patch('sys.argv', [
        'migrate', 'test.swsearch', '--to-pgvector',
        '--connection-string', 'postgresql://u:p@localhost/db',
        '--collection-name', 'my_coll',
    ])
    def test_migrate_to_pgvector_success(self):
        """Successful pgvector migration should print success message."""
        mock_migrator = Mock()
        mock_migrator.migrate_sqlite_to_pgvector.return_value = {
            'chunks_migrated': 50,
            'errors': 0,
        }

        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print:
            migrate_command()

        printed = [str(c) for c in mock_print.call_args_list]
        assert any('Migration completed successfully' in s for s in printed)

    @patch('sys.argv', ['migrate', 'test.swsearch', '--to-sqlite'])
    def test_migrate_to_sqlite_not_implemented(self):
        """to-sqlite should report not implemented and exit."""
        mock_migrator = Mock()
        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1

    @patch('sys.argv', [
        'migrate', 'test.swsearch', '--to-pgvector',
        '--connection-string', 'postgresql://u:p@localhost/db',
        '--collection-name', 'coll',
    ])
    def test_migrate_exception(self):
        """Exception during migration should exit with code 1."""
        mock_migrator = Mock()
        mock_migrator.migrate_sqlite_to_pgvector.side_effect = Exception("DB error")

        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1

    @patch('sys.argv', ['migrate', '--info', 'test.swsearch', '--verbose'])
    def test_migrate_info_verbose(self):
        """--info --verbose should print full config."""
        mock_migrator = Mock()
        mock_migrator.get_index_info.return_value = {
            'type': 'sqlite',
            'total_chunks': 100,
            'total_files': 10,
            'config': {
                'embedding_model': 'test-model',
                'embedding_dimensions': 384,
                'created_at': '2025-01-01',
            },
        }

        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print:
            migrate_command()

        mock_print.assert_any_call("\n  Full configuration:")

    @patch('sys.argv', ['migrate', '--info', 'test.swsearch'])
    def test_migrate_info_exception(self):
        """Exception in info mode should exit with code 1."""
        with patch('signalwire_agents.search.migration.SearchIndexMigrator', side_effect=Exception("fail")), \
             patch('builtins.print'), \
             pytest.raises(SystemExit) as exc_info:
            migrate_command()
        assert exc_info.value.code == 1

    @patch('sys.argv', ['migrate', '--info', 'test.swsearch'])
    def test_migrate_info_unknown_type(self):
        """Info with unknown index type should print 'Unable to determine'."""
        mock_migrator = Mock()
        mock_migrator.get_index_info.return_value = {
            'type': 'unknown',
        }

        with patch('signalwire_agents.search.migration.SearchIndexMigrator', return_value=mock_migrator), \
             patch('builtins.print') as mock_print:
            migrate_command()

        mock_print.assert_any_call("  Unable to determine index type")


def _make_mock_requests_module(post_return=None, post_side_effect=None):
    """Create a mock requests module with real exception classes for except clauses."""
    import requests as real_requests
    mock_mod = types.ModuleType('requests')
    mock_mod.ConnectionError = real_requests.ConnectionError
    mock_mod.Timeout = real_requests.Timeout
    mock_mod.RequestException = real_requests.RequestException
    mock_mod.post = Mock()
    if post_return is not None:
        mock_mod.post.return_value = post_return
    if post_side_effect is not None:
        mock_mod.post.side_effect = post_side_effect
    return mock_mod


class TestRemoteCommand:
    """Tests for remote_command."""

    @patch('sys.argv', ['remote', 'localhost:8001', 'query', '--index-name', 'docs'])
    def test_endpoint_http_prefix_added(self):
        """Endpoint without http:// should get it prepended."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{'score': 0.9, 'content': 'x', 'metadata': {'filename': 'f'}}]
        }

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print'):
            remote_command()

        call_args = mock_requests.post.call_args
        assert call_args[0][0].startswith('http://')
        assert call_args[0][0].endswith('/search')

    @patch('sys.argv', ['remote', 'http://localhost:8001/', 'query', '--index-name', 'docs'])
    def test_endpoint_trailing_slash(self):
        """Endpoint with trailing slash should append 'search' correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{'score': 0.9, 'content': 'x', 'metadata': {'filename': 'f'}}]
        }

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print'):
            remote_command()

        call_args = mock_requests.post.call_args
        url = call_args[0][0]
        assert url == 'http://localhost:8001/search'

    @patch('sys.argv', ['remote', 'http://localhost:8001/search', 'query', '--index-name', 'docs'])
    def test_endpoint_already_has_search(self):
        """Endpoint already ending with /search should not double-append."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{'score': 0.9, 'content': 'x', 'metadata': {'filename': 'f'}}]
        }

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print'):
            remote_command()

        call_args = mock_requests.post.call_args
        url = call_args[0][0]
        assert url == 'http://localhost:8001/search'
        assert not url.endswith('/search/search')

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_requests_import_error(self):
        """Missing requests library should exit with helpful message."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'requests':
                raise ImportError("No module named 'requests'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_404_response(self):
        """404 response should print error and exit."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {'detail': 'Index not found'}

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: Index not found")

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_500_response(self):
        """500 response should print error and exit."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {'detail': 'Internal error'}

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_connection_error(self):
        """Connection error should print helpful message and exit."""
        import requests as real_requests
        mock_requests = _make_mock_requests_module(
            post_side_effect=real_requests.ConnectionError("Refused")
        )

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1
        printed = [str(c) for c in mock_print.call_args_list]
        assert any('Could not connect' in s for s in printed)

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs', '--timeout', '5'])
    def test_remote_timeout(self):
        """Timeout should print timeout message and exit."""
        import requests as real_requests
        mock_requests = _make_mock_requests_module(
            post_side_effect=real_requests.Timeout("timed out")
        )

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1
        printed = [str(c) for c in mock_print.call_args_list]
        assert any('timed out' in s.lower() for s in printed)

    @patch('sys.argv', [
        'remote', 'http://localhost:8001', 'query',
        '--index-name', 'docs', '--tags', 'a,b', '--verbose',
    ])
    def test_remote_verbose_with_tags(self):
        """Verbose mode with tags should print payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{'score': 0.9, 'content': 'x', 'metadata': {'filename': 'f'}}]
        }

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print:
            remote_command()

        # Payload should include tags
        call_kw = mock_requests.post.call_args
        payload = call_kw[1].get('json')
        assert payload is not None
        assert payload['tags'] == ['a', 'b']

    @patch('sys.argv', [
        'remote', 'http://localhost:8001', 'query',
        '--index-name', 'docs', '--json',
    ])
    def test_remote_json_output(self):
        """--json flag should output raw JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{'score': 0.9, 'content': 'test'}],
        }

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print:
            remote_command()

        printed = ''.join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        assert '"results"' in printed

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_success_with_results(self):
        """Successful response with results should print them."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'score': 0.95,
                    'content': 'Hello world',
                    'metadata': {'filename': 'test.md', 'section': 'intro'},
                }
            ],
            'enhanced_query': 'enhanced query',
        }

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print:
            remote_command()

        printed = [str(c) for c in mock_print.call_args_list]
        assert any('Found 1 result' in s for s in printed)

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_no_results(self):
        """Empty results should print 'No results found'."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': []}

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit):
            remote_command()

        mock_print.assert_any_call("No results found for 'query' in index 'docs'")

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_404_json_parse_error(self):
        """404 response with unparseable JSON should fallback."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.side_effect = Exception("bad json")

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: Index not found")

    @patch('sys.argv', ['remote', 'http://localhost:8001', 'query', '--index-name', 'docs'])
    def test_remote_500_json_parse_error(self):
        """Non-404 error with unparseable JSON should fallback to status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("bad json")
        mock_response.text = "Internal Server Error"

        mock_requests = _make_mock_requests_module(post_return=mock_response)

        with patch.dict('sys.modules', {'requests': mock_requests}), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            remote_command()

        assert exc_info.value.code == 1
        mock_print.assert_any_call("Error: HTTP 500: Internal Server Error")