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