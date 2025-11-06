"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import nltk
import re
from typing import Dict, Any, List, Optional
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to track if we've already warned about spaCy
_spacy_warning_shown = False

# Language detection and spaCy model loading
def detect_language(text: str) -> str:
    """
    Detect language of input text
    Simple implementation - can be enhanced with langdetect library
    """
    # Simple heuristic-based detection
    # In a full implementation, you'd use langdetect or similar
    common_english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
    common_spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'han', 'fue', 'ser', 'está', 'todo', 'más', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'donde', 'quien', 'desde', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas'}
    
    words = text.lower().split()
    english_count = sum(1 for word in words if word in common_english_words)
    spanish_count = sum(1 for word in words if word in common_spanish_words)
    
    if spanish_count > english_count:
        return 'es'
    else:
        return 'en'

def load_spacy_model(language: str):
    """
    Load spaCy model for the given language
    Returns None if spaCy is not available or model not found
    """
    global _spacy_warning_shown
    
    try:
        import spacy
        
        # Language model mapping
        model_map = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm'
        }
        
        model_name = model_map.get(language, 'en_core_web_sm')
        
        try:
            return spacy.load(model_name)
        except OSError:
            if not _spacy_warning_shown:
                logger.warning(f"spaCy model '{model_name}' not found. Falling back to NLTK.")
                _spacy_warning_shown = True
            return None
            
    except ImportError:
        if not _spacy_warning_shown:
            logger.warning("spaCy not available. Using NLTK for POS tagging.")
            _spacy_warning_shown = True
        return None

# Model cache - stores multiple models by name
_model_cache = {}  # model_name -> SentenceTransformer instance
_model_lock = None

def set_global_model(model):
    """Legacy function - adds model to cache instead of setting globally"""
    if model and hasattr(model, 'model_name'):
        _model_cache[model.model_name] = model
        logger.info(f"Model added to cache: {model.model_name}")

def _get_cached_model(model_name: str = None):
    """Get or create cached sentence transformer model

    Args:
        model_name: Optional model name. If not provided, uses default.
    """
    global _model_cache, _model_lock

    # Default model
    if model_name is None:
        model_name = 'sentence-transformers/all-mpnet-base-v2'

    # Initialize lock if needed
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    # Check if model is already in cache
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Load model with lock to prevent race conditions
    with _model_lock:
        # Double check in case another thread loaded it
        if model_name in _model_cache:
            return _model_cache[model_name]

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer model: {model_name}")
            model = SentenceTransformer(model_name)
            # Store the model name for identification
            model.model_name = model_name
            # Add to cache
            _model_cache[model_name] = model
            logger.info(f"Successfully loaded and cached model: {model_name}")
            return model
        except ImportError:
            logger.error("sentence-transformers not available. Cannot load model.")
            return None
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

def vectorize_query(query: str, model=None, model_name: str = None):
    """
    Vectorize query using sentence transformers
    Returns numpy array of embeddings
    
    Args:
        query: Query string to vectorize
        model: Optional pre-loaded model instance. If not provided, uses cached model.
        model_name: Optional model name to use if loading a new model
    """
    try:
        import numpy as np
        
        # Use provided model or get cached one
        if model is None:
            model = _get_cached_model(model_name)
            if model is None:
                return None
        
        embedding = model.encode(query, show_progress_bar=False)
        return embedding
        
    except ImportError:
        logger.error("numpy not available. Cannot vectorize query.")
        return None
    except Exception as e:
        logger.error(f"Error vectorizing query: {e}")
        return None

# Language to NLTK stopwords mapping
stopwords_language_map = {
    'en': 'english',
    'es': 'spanish', 
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'nl': 'dutch',
    'ru': 'russian',
    'ar': 'arabic',
    'da': 'danish',
    'fi': 'finnish',
    'hu': 'hungarian',
    'no': 'norwegian',
    'ro': 'romanian',
    'sv': 'swedish',
    'tr': 'turkish'
}

# Function to ensure NLTK resources are downloaded
def ensure_nltk_resources():
    """Download required NLTK resources if not already present"""
    resources = ['punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']
    for resource in resources:
        try:
            # Try different paths for different resource types
            if resource in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{resource}')
            elif resource in ['wordnet']:
                nltk.data.find(f'corpora/{resource}')
            elif resource in ['averaged_perceptron_tagger']:
                nltk.data.find(f'taggers/{resource}')
            elif resource in ['stopwords']:
                nltk.data.find(f'corpora/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource '{resource}'")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource '{resource}': {e}")
                # Continue without this resource - some functionality may be degraded

# Initialize NLTK resources
ensure_nltk_resources()

# Mapping spaCy POS tags to WordNet POS tags
pos_mapping = {
    'NOUN': wn.NOUN,
    'VERB': wn.VERB,
    'ADJ': wn.ADJ,
    'ADV': wn.ADV,
    'PROPN': wn.NOUN,  # Proper nouns as nouns
}

def get_wordnet_pos(spacy_pos):
    """Map spaCy POS tags to WordNet POS tags."""
    return pos_mapping.get(spacy_pos, wn.NOUN)

def get_synonyms(word: str, pos_tag: str, max_synonyms: int = 5) -> List[str]:
    """Get synonyms for a word using WordNet"""
    try:
        wn_pos = get_wordnet_pos(pos_tag)
        synsets = wn.synsets(word, pos=wn_pos)
        synonyms = set()
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                synonyms.add(synonym.lower())
                if len(synonyms) >= max_synonyms:
                    break
            if len(synonyms) >= max_synonyms:
                break
        return list(synonyms)
    except Exception as e:
        logger.warning(f"Error getting synonyms for '{word}': {e}")
        return []

def remove_duplicate_words(input_string: str) -> str:
    """Remove duplicate words from the input string while preserving the order and punctuation."""
    words = re.findall(r'\b\w+\b', input_string)
    seen = set()
    result = []

    for word in words:
        if word.lower() not in seen:
            seen.add(word.lower())
            result.append(word)

    words_with_punctuation = input_string.split()
    final_result = []
    for word in words_with_punctuation:
        clean_word = re.sub(r'\W+', '', word)
        if clean_word.lower() in seen:
            final_result.append(word)
            seen.remove(clean_word.lower())

    return ' '.join(final_result)

def preprocess_query(query: str, language: str = 'en', pos_to_expand: Optional[List[str]] = None, 
                    max_synonyms: int = 5, debug: bool = False, vector: bool = False, 
                    vectorize_query_param: bool = False, nlp_backend: str = None, 
                    query_nlp_backend: str = 'nltk', model_name: str = None,
                    preserve_original: bool = True) -> Dict[str, Any]:
    """
    Advanced query preprocessing with language detection, POS tagging, synonym expansion, and vectorization
    
    Args:
        query: Input query string
        language: Language code ('en', 'es', etc.) or 'auto' for detection
        pos_to_expand: List of POS tags to expand with synonyms
        max_synonyms: Maximum synonyms per word
        debug: Enable debug output
        vector: Include vector embedding in output
        vectorize_query_param: If True, just vectorize without other processing
        nlp_backend: DEPRECATED - use query_nlp_backend instead
        query_nlp_backend: NLP backend for query processing ('nltk' for fast, 'spacy' for better quality)
        
    Returns:
        Dict containing processed query, language, POS tags, and optionally vector
    """
    
    # Handle backward compatibility
    if nlp_backend is not None:
        query_nlp_backend = nlp_backend
        if debug:
            logger.info(f"Using deprecated 'nlp_backend' parameter, please use 'query_nlp_backend' instead")
    
    if vectorize_query_param:
        # Vectorize the query directly
        vectorized_query = vectorize_query(query)
        if vectorized_query is not None:
            return {
                'input': query,
                'vector': vectorized_query.tolist()
            }
        else:
            return {'input': query, 'vector': None}

    if pos_to_expand is None:
        pos_to_expand = ['NOUN', 'VERB', 'ADJ']  # Default to expanding synonyms for nouns, verbs, and adjectives

    # Detect language if set to 'auto'
    if language == 'auto':
        language = detect_language(query)
        if debug:
            logger.info(f"Detected language: {language}")
    
    # Load spaCy model based on the language and backend choice
    nlp = None
    if query_nlp_backend == 'spacy':
        nlp = load_spacy_model(language)
        if nlp is None and debug:
            logger.info("spaCy backend requested but not available, falling back to NLTK")
    elif query_nlp_backend == 'nltk':
        if debug:
            logger.info("Using NLTK backend for query processing")
    else:
        logger.warning(f"Unknown query NLP backend '{query_nlp_backend}', using NLTK")
        query_nlp_backend = 'nltk'
    
    # Tokenization and stop word removal
    try:
        tokens = nltk.word_tokenize(query)
    except LookupError as e:
        # If tokenization fails, try to download punkt resources
        logger.warning(f"NLTK tokenization failed: {e}")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            tokens = nltk.word_tokenize(query)
        except Exception as fallback_error:
            # If all else fails, use simple split as fallback
            logger.warning(f"NLTK tokenization fallback failed: {fallback_error}. Using simple word splitting.")
            tokens = query.split()
    
    nltk_language = stopwords_language_map.get(language, 'english')
    
    try:
        stop_words = set(nltk.corpus.stopwords.words(nltk_language))
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
            stop_words = set(nltk.corpus.stopwords.words(nltk_language))
        except:
            logger.warning(f"Could not load stopwords for language '{nltk_language}', using English")
            stop_words = set(nltk.corpus.stopwords.words('english'))
    
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization and POS Tagging using spaCy or NLTK
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = PorterStemmer()
    lemmas = []
    pos_tags = {}

    if nlp and query_nlp_backend == 'spacy':
        # Use spaCy for better POS tagging
        doc = nlp(" ".join(tokens))
        for token in doc:
            lemma = token.lemma_.lower()
            stemmed = stemmer.stem(lemma)
            lemmas.append((token.text.lower(), stemmed))
            pos_tags[token.text.lower()] = token.pos_
        if debug:
            logger.info(f"POS Tagging Results (spaCy): {pos_tags}")
    else:
        # Use NLTK (default or fallback)
        try:
            nltk_pos_tags = nltk.pos_tag(tokens)
            for token, pos_tag in nltk_pos_tags:
                try:
                    lemma = lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)).lower()
                except Exception:
                    # Fallback if lemmatization fails
                    lemma = token.lower()
                stemmed = stemmer.stem(lemma)
                lemmas.append((token.lower(), stemmed))
                pos_tags[token.lower()] = pos_tag
            if debug:
                logger.info(f"POS Tagging Results (NLTK): {pos_tags}")
        except Exception as pos_error:
            # Fallback if POS tagging fails completely
            logger.warning(f"NLTK POS tagging failed: {pos_error}. Using basic token processing.")
            for token in tokens:
                lemma = token.lower()
                stemmed = stemmer.stem(lemma)
                lemmas.append((token.lower(), stemmed))
                pos_tags[token.lower()] = 'NN'  # Default to noun
            if debug:
                logger.info(f"Using fallback token processing for: {tokens}")

    # Expanding query with synonyms
    expanded_query_set = set()
    expanded_query = []
    
    # If preserve_original is True, always include the original query first
    if preserve_original:
        # Add original query terms first (maintains exact phrases)
        original_tokens = query.lower().split()
        for token in original_tokens:
            if token not in expanded_query_set:
                expanded_query.append(token)
                expanded_query_set.add(token)
    
    for original, lemma in lemmas:
        if original not in expanded_query_set:
            expanded_query.append(original)
            expanded_query_set.add(original)
        if lemma not in expanded_query_set and not preserve_original:  # Only add lemmas if not preserving original
            expanded_query.append(lemma)
            expanded_query_set.add(lemma)
        if pos_tags.get(original) in pos_to_expand and max_synonyms > 0:
            synonyms = get_synonyms(lemma, pos_tags[original], max_synonyms)
            for synonym in synonyms:
                if synonym not in expanded_query_set:
                    expanded_query.append(synonym)
                    expanded_query_set.add(synonym)
    
    # Convert to array, remove duplicates, and join back to string
    final_query_str = " ".join(expanded_query)
    final_query_str = remove_duplicate_words(final_query_str)

    if debug:
        logger.info(f"Expanded Query: {final_query_str}")
        logger.info(f"NLP Backend Used: {query_nlp_backend if nlp or query_nlp_backend == 'nltk' else 'nltk (fallback)'}")
    
    formatted_output = {
        'input': final_query_str,
        'enhanced_text': final_query_str,  # Alias for compatibility
        'language': language,
        'POS': pos_tags,
        'nlp_backend_used': query_nlp_backend if nlp or query_nlp_backend == 'nltk' else 'nltk'
    }
    
    # Vectorize query if requested
    if vector:
        vectorized_query = vectorize_query(final_query_str, model_name=model_name)
        if vectorized_query is not None:
            formatted_output['vector'] = vectorized_query.tolist()
        else:
            formatted_output['vector'] = None

    return formatted_output

def preprocess_document_content(content: str, language: str = 'en', nlp_backend: str = None, 
                               index_nlp_backend: str = 'nltk') -> Dict[str, Any]:
    """
    Preprocess document content for better searchability
    
    Args:
        content: Document content to process
        language: Language code for processing
        nlp_backend: DEPRECATED - use index_nlp_backend instead
        index_nlp_backend: NLP backend for document processing ('nltk' for fast, 'spacy' for better quality)
        
    Returns:
        Dict containing enhanced text and extracted keywords
    """
    
    # Handle backward compatibility
    if nlp_backend is not None:
        index_nlp_backend = nlp_backend
    
    # Use existing preprocessing but adapted for documents
    processed = preprocess_query(
        content,
        language=language,
        pos_to_expand=['NOUN', 'VERB'],  # Less aggressive for documents
        max_synonyms=2,  # Fewer synonyms for documents
        debug=False,
        vector=False,
        query_nlp_backend=index_nlp_backend
    )
    
    # Extract key terms for keyword search
    try:
        tokens = nltk.word_tokenize(processed['input'])
        nltk_language = stopwords_language_map.get(language, 'english')
        
        try:
            stop_words = set(nltk.corpus.stopwords.words(nltk_language))
        except:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            
        keywords = [word.lower() for word in tokens if word.lower() not in stop_words and len(word) > 2]
        
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        keywords = []
    
    return {
        'enhanced_text': processed['input'],
        'keywords': keywords[:20],  # Limit to top 20 keywords
        'language': processed.get('language', language),
        'pos_analysis': processed.get('POS', {})
    } 