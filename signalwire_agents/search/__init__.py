"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""SignalWire Agents Local Search Module

This module provides local search capabilities for the SignalWire Agents SDK.
It requires additional dependencies that can be installed with:

    pip install signalwire-agents[search]           # Basic search
    pip install signalwire-agents[search-full]      # + Document processing
    pip install signalwire-agents[search-nlp]       # + Advanced NLP
    pip install signalwire-agents[search-all]       # All features
"""

import warnings

# Check for core search dependencies
_SEARCH_AVAILABLE = True
_MISSING_DEPS = []

try:
    import numpy
except ImportError:
    _SEARCH_AVAILABLE = False
    _MISSING_DEPS.append('numpy')

try:
    import sklearn
except ImportError:
    _SEARCH_AVAILABLE = False
    _MISSING_DEPS.append('scikit-learn')

try:
    import sentence_transformers
except ImportError:
    _SEARCH_AVAILABLE = False
    _MISSING_DEPS.append('sentence-transformers')

try:
    import nltk
except ImportError:
    _SEARCH_AVAILABLE = False
    _MISSING_DEPS.append('nltk')

def _check_search_dependencies():
    """Check if search dependencies are available and provide helpful error message"""
    if not _SEARCH_AVAILABLE:
        missing = ', '.join(_MISSING_DEPS)
        raise ImportError(
            f"Search functionality requires additional dependencies: {missing}\n"
            f"Install with: pip install signalwire-agents[search]\n"
            f"For full features: pip install signalwire-agents[search-all]"
        )

# Conditional imports based on available dependencies
__all__ = []

if _SEARCH_AVAILABLE:
    try:
        from .query_processor import preprocess_query, preprocess_document_content
        from .document_processor import DocumentProcessor
        from .index_builder import IndexBuilder
        from .search_engine import SearchEngine
        from .search_service import SearchService
        from .models import MODEL_ALIASES, DEFAULT_MODEL, resolve_model_alias
        from .migration import SearchIndexMigrator
        
        __all__ = [
            'preprocess_query',
            'preprocess_document_content', 
            'DocumentProcessor',
            'IndexBuilder',
            'SearchEngine',
            'SearchService',
            'MODEL_ALIASES',
            'DEFAULT_MODEL',
            'resolve_model_alias',
            'SearchIndexMigrator'
        ]
    except ImportError as e:
        # Some search components failed to import
        warnings.warn(
            f"Some search components failed to import: {e}\n"
            f"For full search functionality, install: pip install signalwire-agents[search-all]",
            ImportWarning
        )
        
        # Try to import what we can
        try:
            from .query_processor import preprocess_query, preprocess_document_content
            __all__.extend(['preprocess_query', 'preprocess_document_content'])
        except ImportError:
            pass
            
        try:
            from .document_processor import DocumentProcessor
            __all__.append('DocumentProcessor')
        except ImportError:
            pass
else:
    # Provide stub functions that give helpful error messages
    def preprocess_query(*args, **kwargs):
        _check_search_dependencies()
    
    def preprocess_document_content(*args, **kwargs):
        _check_search_dependencies()
    
    class DocumentProcessor:
        def __init__(self, *args, **kwargs):
            _check_search_dependencies()
    
    class IndexBuilder:
        def __init__(self, *args, **kwargs):
            _check_search_dependencies()
    
    class SearchEngine:
        def __init__(self, *args, **kwargs):
            _check_search_dependencies()
    
    class SearchService:
        def __init__(self, *args, **kwargs):
            _check_search_dependencies()
    
    __all__ = [
        'preprocess_query',
        'preprocess_document_content', 
        'DocumentProcessor',
        'IndexBuilder',
        'SearchEngine',
        'SearchService'
    ] 