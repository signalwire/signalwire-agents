"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for search query processor module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from signalwire_agents.search.query_processor import (
    detect_language,
    load_spacy_model,
    vectorize_query,
    ensure_nltk_resources,
    get_wordnet_pos,
    get_synonyms,
    remove_duplicate_words,
    preprocess_query,
    preprocess_document_content
)


class TestLanguageDetection:
    """Test language detection functionality"""
    
    def test_detect_english(self):
        """Test detection of English text"""
        english_text = "The quick brown fox jumps over the lazy dog"
        assert detect_language(english_text) == 'en'
    
    def test_detect_spanish(self):
        """Test detection of Spanish text"""
        spanish_text = "El perro come la comida en el parque"
        assert detect_language(spanish_text) == 'es'
    
    def test_detect_mixed_language_english_dominant(self):
        """Test detection when English words dominate"""
        mixed_text = "The dog and el gato are friends"
        assert detect_language(mixed_text) == 'en'
    
    def test_detect_mixed_language_spanish_dominant(self):
        """Test detection when Spanish words dominate"""
        mixed_text = "El perro y the cat son amigos"
        assert detect_language(mixed_text) == 'es'
    
    def test_detect_empty_text(self):
        """Test detection with empty text"""
        assert detect_language("") == 'en'  # Default to English
    
    def test_detect_unknown_language(self):
        """Test detection with unknown language defaults to English"""
        unknown_text = "xyz abc def ghi"
        assert detect_language(unknown_text) == 'en'


class TestSpacyModelLoading:
    """Test spaCy model loading functionality"""
    
    def test_load_spacy_model_success(self):
        """Test successful spaCy model loading"""
        with patch('builtins.__import__') as mock_import:
            mock_spacy = Mock()
            mock_model = Mock()
            mock_spacy.load.return_value = mock_model
            mock_import.return_value = mock_spacy
            
            result = load_spacy_model('en')
            
            mock_spacy.load.assert_called_once_with('en_core_web_sm')
            assert result == mock_model
    
    def test_load_spacy_model_not_found(self):
        """Test spaCy model loading when model not found"""
        with patch('builtins.__import__') as mock_import:
            mock_spacy = Mock()
            mock_spacy.load.side_effect = OSError("Model not found")
            mock_import.return_value = mock_spacy
            
            with patch('signalwire_agents.search.query_processor.logger') as mock_logger:
                result = load_spacy_model('en')
                
                assert result is None
                mock_logger.warning.assert_called_once()
    
    def test_load_spacy_model_import_error(self):
        """Test spaCy model loading when spaCy not available"""
        with patch('builtins.__import__', side_effect=ImportError()):
            with patch('signalwire_agents.search.query_processor.logger') as mock_logger:
                # Reset the global warning flag to ensure warning is shown
                with patch('signalwire_agents.search.query_processor._spacy_warning_shown', False):
                    result = load_spacy_model('en')
                    
                    assert result is None
                    mock_logger.warning.assert_called_once()
    
    def test_load_spacy_model_different_languages(self):
        """Test loading models for different languages"""
        with patch('builtins.__import__') as mock_import:
            mock_spacy = Mock()
            mock_model = Mock()
            mock_spacy.load.return_value = mock_model
            mock_import.return_value = mock_spacy
            
            # Test various languages
            languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
            expected_models = [
                'en_core_web_sm', 'es_core_news_sm', 'fr_core_news_sm',
                'de_core_news_sm', 'it_core_news_sm', 'pt_core_news_sm'
            ]
            
            for lang, expected_model in zip(languages, expected_models):
                load_spacy_model(lang)
                mock_spacy.load.assert_called_with(expected_model)
    
    def test_load_spacy_model_unknown_language(self):
        """Test loading model for unknown language defaults to English"""
        with patch('builtins.__import__') as mock_import:
            mock_spacy = Mock()
            mock_model = Mock()
            mock_spacy.load.return_value = mock_model
            mock_import.return_value = mock_spacy
            
            load_spacy_model('unknown')
            mock_spacy.load.assert_called_with('en_core_web_sm')


class TestQueryVectorization:
    """Test query vectorization functionality"""
    
    def test_vectorize_query_success(self):
        """Test successful query vectorization"""
        from signalwire_agents.search import query_processor as _qp

        mock_model = Mock()
        mock_embedding = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding

        # Clear model cache so _get_cached_model loads a fresh model
        _qp._model_cache.clear()

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model) as mock_st:
            result = vectorize_query("test query")

            mock_st.assert_called_once_with('sentence-transformers/all-mpnet-base-v2')
            mock_model.encode.assert_called_once_with("test query", show_progress_bar=False)
            assert result == mock_embedding

        # Clean up cached mock model
        _qp._model_cache.clear()
    
    def test_vectorize_query_import_error(self):
        """Test query vectorization when sentence-transformers not available"""
        with patch('builtins.__import__', side_effect=ImportError()):
            with patch('signalwire_agents.search.query_processor.logger') as mock_logger:
                result = vectorize_query("test query")
                
                assert result is None
                mock_logger.error.assert_called_once()


class TestNLTKResources:
    """Test NLTK resource management"""
    
    @patch('signalwire_agents.search.query_processor.nltk')
    def test_ensure_nltk_resources_already_present(self, mock_nltk):
        """Test when NLTK resources are already present"""
        mock_nltk.data.find.return_value = True
        
        ensure_nltk_resources()
        
        # Should not call download if resources are found
        mock_nltk.download.assert_not_called()
    
    @patch('signalwire_agents.search.query_processor.nltk')
    def test_ensure_nltk_resources_download_needed(self, mock_nltk):
        """Test when NLTK resources need to be downloaded"""
        mock_nltk.data.find.side_effect = LookupError("Resource not found")
        
        ensure_nltk_resources()
        
        # Should call download for each missing resource
        assert mock_nltk.download.call_count == 5  # punkt, punkt_tab, wordnet, averaged_perceptron_tagger, stopwords
    
    @patch('signalwire_agents.search.query_processor.nltk')
    def test_ensure_nltk_resources_download_error(self, mock_nltk):
        """Test when NLTK resource download fails"""
        mock_nltk.data.find.side_effect = LookupError("Resource not found")
        mock_nltk.download.side_effect = Exception("Download failed")
        
        with patch('signalwire_agents.search.query_processor.logger') as mock_logger:
            ensure_nltk_resources()
            
            # Should log warnings for failed downloads
            assert mock_logger.warning.call_count == 5


class TestWordNetUtilities:
    """Test WordNet utility functions"""
    
    def test_get_wordnet_pos_known_tags(self):
        """Test mapping of known POS tags"""
        from nltk.corpus import wordnet as wn
        
        assert get_wordnet_pos('NOUN') == wn.NOUN
        assert get_wordnet_pos('VERB') == wn.VERB
        assert get_wordnet_pos('ADJ') == wn.ADJ
        assert get_wordnet_pos('ADV') == wn.ADV
        assert get_wordnet_pos('PROPN') == wn.NOUN
    
    def test_get_wordnet_pos_unknown_tag(self):
        """Test mapping of unknown POS tag defaults to NOUN"""
        from nltk.corpus import wordnet as wn
        
        assert get_wordnet_pos('UNKNOWN') == wn.NOUN
    
    @patch('signalwire_agents.search.query_processor.wn')
    def test_get_synonyms_success(self, mock_wn):
        """Test successful synonym retrieval"""
        # Mock WordNet synsets and lemmas
        mock_lemma1 = Mock()
        mock_lemma1.name.return_value = 'happy'
        mock_lemma2 = Mock()
        mock_lemma2.name.return_value = 'joyful'
        
        mock_synset = Mock()
        mock_synset.lemmas.return_value = [mock_lemma1, mock_lemma2]
        
        mock_wn.synsets.return_value = [mock_synset]
        mock_wn.NOUN = 'n'
        
        result = get_synonyms('glad', 'NOUN', max_synonyms=5)
        
        assert 'happy' in result
        assert 'joyful' in result
    
    @patch('signalwire_agents.search.query_processor.wn')
    def test_get_synonyms_no_synsets(self, mock_wn):
        """Test synonym retrieval when no synsets found"""
        mock_wn.synsets.return_value = []
        mock_wn.NOUN = 'n'
        
        result = get_synonyms('nonexistentword', 'NOUN')
        
        assert result == []
    
    @patch('signalwire_agents.search.query_processor.wn')
    def test_get_synonyms_error(self, mock_wn):
        """Test synonym retrieval with error"""
        mock_wn.synsets.side_effect = Exception("WordNet error")
        
        with patch('signalwire_agents.search.query_processor.logger') as mock_logger:
            result = get_synonyms('word', 'NOUN')
            
            assert result == []
            mock_logger.warning.assert_called_once()


class TestTextProcessing:
    """Test text processing utilities"""
    
    def test_remove_duplicate_words_basic(self):
        """Test basic duplicate word removal"""
        text = "the quick brown fox jumps over the lazy dog"
        result = remove_duplicate_words(text)
        
        # Should remove the second "the"
        words = result.split()
        assert words.count('the') == 1
        assert 'quick' in words
        assert 'brown' in words
    
    def test_remove_duplicate_words_with_punctuation(self):
        """Test duplicate removal with punctuation"""
        text = "Hello, world! Hello again."
        result = remove_duplicate_words(text)
        
        # Should preserve punctuation but remove duplicate "Hello"
        assert result.count('Hello') == 1
        assert ',' in result or '!' in result
    
    def test_remove_duplicate_words_case_insensitive(self):
        """Test duplicate removal is case insensitive"""
        text = "The cat and THE dog"
        result = remove_duplicate_words(text)
        
        # Should remove duplicate "THE" (case insensitive)
        words = result.lower().split()
        assert words.count('the') == 1
    
    def test_remove_duplicate_words_empty_string(self):
        """Test duplicate removal with empty string"""
        result = remove_duplicate_words("")
        assert result == ""
    
    def test_remove_duplicate_words_no_duplicates(self):
        """Test duplicate removal when no duplicates exist"""
        text = "unique words only here"
        result = remove_duplicate_words(text)
        assert result == text


class TestQueryPreprocessing:
    """Test query preprocessing functionality"""
    
    @patch('signalwire_agents.search.query_processor.detect_language')
    @patch('signalwire_agents.search.query_processor.load_spacy_model')
    def test_preprocess_query_basic(self, mock_load_spacy, mock_detect_lang):
        """Test basic query preprocessing"""
        mock_detect_lang.return_value = 'en'
        mock_load_spacy.return_value = None  # Use NLTK fallback
        
        result = preprocess_query("test query")
        
        assert 'enhanced_text' in result
        assert 'language' in result
        assert result['language'] == 'en'
    
    @patch('signalwire_agents.search.query_processor.detect_language')
    @patch('signalwire_agents.search.query_processor.vectorize_query')
    def test_preprocess_query_with_vector(self, mock_vectorize, mock_detect_lang):
        """Test query preprocessing with vectorization"""
        mock_detect_lang.return_value = 'en'
        # Mock vectorize_query to return a numpy-like object with tolist method
        mock_vector = Mock()
        mock_vector.tolist.return_value = [0.1, 0.2, 0.3]
        mock_vectorize.return_value = mock_vector
        
        result = preprocess_query("test query", vector=True)
        
        assert 'vector' in result
        assert result['vector'] == [0.1, 0.2, 0.3]
        mock_vectorize.assert_called_once()
    
    @patch('signalwire_agents.search.query_processor.vectorize_query')
    def test_preprocess_query_vectorize_only(self, mock_vectorize):
        """Test query preprocessing with vectorize_query_param=True"""
        # Mock vectorize_query to return a numpy-like object with tolist method
        mock_vector = Mock()
        mock_vector.tolist.return_value = [0.1, 0.2, 0.3]
        mock_vectorize.return_value = mock_vector
        
        result = preprocess_query("test query", vectorize_query_param=True)
        
        assert 'vector' in result
        assert result['vector'] == [0.1, 0.2, 0.3]
        mock_vectorize.assert_called_once_with("test query")
    
    def test_preprocess_query_auto_language_detection(self):
        """Test query preprocessing with automatic language detection"""
        with patch('signalwire_agents.search.query_processor.detect_language') as mock_detect:
            mock_detect.return_value = 'es'
            
            result = preprocess_query("hola mundo", language='auto')
            
            mock_detect.assert_called_once_with("hola mundo")
            assert result['language'] == 'es'
    
    @patch('signalwire_agents.search.query_processor.load_spacy_model')
    def test_preprocess_query_with_spacy(self, mock_load_spacy):
        """Test query preprocessing with spaCy backend"""
        # Mock spaCy model and processing
        mock_doc = Mock()
        mock_token1 = Mock()
        mock_token1.text = 'test'
        mock_token1.pos_ = 'NOUN'
        mock_token1.lemma_ = 'test'
        
        mock_token2 = Mock()
        mock_token2.text = 'query'
        mock_token2.pos_ = 'NOUN'
        mock_token2.lemma_ = 'query'
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_load_spacy.return_value = mock_nlp
        
        result = preprocess_query("test query", nlp_backend='spacy')
        
        assert 'enhanced_text' in result
        mock_load_spacy.assert_called_once()
    
    @patch('signalwire_agents.search.query_processor.get_synonyms')
    @patch('signalwire_agents.search.query_processor.load_spacy_model')
    def test_preprocess_query_with_synonym_expansion(self, mock_load_spacy, mock_get_synonyms):
        """Test query preprocessing with synonym expansion"""
        mock_load_spacy.return_value = None  # Use NLTK
        mock_get_synonyms.return_value = ['happy', 'joyful']
        
        result = preprocess_query("glad", pos_to_expand=['NN'], max_synonyms=2)
        
        assert 'enhanced_text' in result
        # Should include original word and synonyms
        enhanced_text = result['enhanced_text']
        assert 'glad' in enhanced_text


class TestDocumentPreprocessing:
    """Test document content preprocessing"""
    
    @patch('signalwire_agents.search.query_processor.load_spacy_model')
    def test_preprocess_document_content_basic(self, mock_load_spacy):
        """Test basic document content preprocessing"""
        mock_load_spacy.return_value = None  # Use NLTK fallback
        
        content = "This is a test document with some content."
        result = preprocess_document_content(content)
        
        assert 'enhanced_text' in result
        assert 'language' in result
        assert result['language'] == 'en'
        assert 'keywords' in result
    
    @patch('signalwire_agents.search.query_processor.load_spacy_model')
    def test_preprocess_document_content_with_spacy(self, mock_load_spacy):
        """Test document preprocessing with spaCy backend"""
        # Mock spaCy processing
        mock_doc = Mock()
        mock_token = Mock()
        mock_token.text = 'test'
        mock_token.pos_ = 'NOUN'
        mock_token.lemma_ = 'test'
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_load_spacy.return_value = mock_nlp
        
        result = preprocess_document_content("test content", nlp_backend='spacy')
        
        assert 'enhanced_text' in result
        assert 'keywords' in result
        mock_load_spacy.assert_called_once()
    
    def test_preprocess_document_content_different_language(self):
        """Test document preprocessing with different language"""
        content = "Este es un documento de prueba."
        result = preprocess_document_content(content, language='es')
        
        assert result['language'] == 'es'
        assert 'enhanced_text' in result
        assert 'keywords' in result
    
    def test_preprocess_document_content_empty(self):
        """Test document preprocessing with empty content"""
        result = preprocess_document_content("")
        
        assert result['enhanced_text'] == ""
        assert result['language'] == 'en'
        assert 'keywords' in result


class TestErrorHandling:
    """Test error handling in query processor"""
    
    @patch('signalwire_agents.search.query_processor.nltk')
    def test_preprocess_query_nltk_error(self, mock_nltk):
        """Test query preprocessing when NLTK operations fail gracefully"""
        # Mock NLTK to work for most operations but fail for one
        mock_nltk.word_tokenize.return_value = ['test', 'query']
        mock_nltk.corpus.stopwords.words.return_value = []
        mock_nltk.pos_tag.return_value = [('test', 'NN'), ('query', 'NN')]
        mock_nltk.WordNetLemmatizer.return_value.lemmatize.return_value = 'test'
        
        # Should not crash, should handle gracefully
        result = preprocess_query("test query")
        
        assert 'enhanced_text' in result
        # Should fallback to basic processing
    
    @patch('signalwire_agents.search.query_processor.vectorize_query')
    def test_preprocess_query_vectorization_error(self, mock_vectorize):
        """Test query preprocessing when vectorization fails"""
        mock_vectorize.return_value = None  # Simulate vectorization failure
        
        result = preprocess_query("test query", vector=True)
        
        # Should handle error gracefully
        assert 'enhanced_text' in result
        assert result.get('vector') is None 