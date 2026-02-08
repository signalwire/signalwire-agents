"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for search document processor module
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

from signalwire_agents.search.document_processor import DocumentProcessor


class TestDocumentProcessorInit:
    """Test DocumentProcessor initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        processor = DocumentProcessor()
        
        assert processor.chunking_strategy == 'sentence'
        assert processor.max_sentences_per_chunk == 5
        assert processor.chunk_size == 50
        assert processor.chunk_overlap == 10
        assert processor.split_newlines is None
        assert processor.chunk_overlap == 10  # Legacy support
        assert processor.semantic_threshold == 0.5
        assert processor.topic_threshold == 0.3
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        processor = DocumentProcessor(
            chunking_strategy='sliding',
            max_sentences_per_chunk=25,
            chunk_size=100,
            chunk_overlap=20,
            split_newlines=3
        )
        
        assert processor.chunking_strategy == 'sliding'
        assert processor.max_sentences_per_chunk == 25
        assert processor.chunk_size == 100
        assert processor.chunk_overlap == 20
        assert processor.split_newlines == 3
        assert processor.chunk_overlap == 20
        assert processor.semantic_threshold == 0.5
        assert processor.topic_threshold == 0.3


class TestDocumentProcessorChunking:
    """Test document chunking strategies"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor()
        self.sample_text = "This is the first sentence. This is the second sentence. This is the third sentence."
        self.filename = "test.txt"
        self.file_type = "txt"
    
    def test_create_chunks_sentence_strategy(self):
        """Test create_chunks with sentence strategy"""
        processor = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=2)
        
        chunks = processor.create_chunks(self.sample_text, self.filename, self.file_type)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('filename' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all(chunk['metadata']['chunk_method'] == 'sentence_based' for chunk in chunks)
    
    def test_create_chunks_sliding_strategy(self):
        """Test create_chunks with sliding window strategy"""
        processor = DocumentProcessor(chunking_strategy='sliding', chunk_size=5, chunk_overlap=2)
        
        chunks = processor.create_chunks(self.sample_text, self.filename, self.file_type)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all(chunk['metadata']['chunk_method'] == 'sliding_window' for chunk in chunks)
        assert all('chunk_size_words' in chunk['metadata'] for chunk in chunks)
        assert all('overlap_size_words' in chunk['metadata'] for chunk in chunks)
    
    def test_create_chunks_paragraph_strategy(self):
        """Test create_chunks with paragraph strategy"""
        processor = DocumentProcessor(chunking_strategy='paragraph')
        paragraph_text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        chunks = processor.create_chunks(paragraph_text, self.filename, self.file_type)
        
        assert len(chunks) == 3
        assert all(chunk['metadata']['chunk_method'] == 'paragraph_based' for chunk in chunks)
        assert chunks[0]['content'] == "First paragraph."
        assert chunks[1]['content'] == "Second paragraph."
        assert chunks[2]['content'] == "Third paragraph."
    
    def test_create_chunks_page_strategy(self):
        """Test create_chunks with page strategy"""
        processor = DocumentProcessor(chunking_strategy='page')
        
        # Test with list input (like PDF pages)
        page_list = ["Page 1 content", "Page 2 content", "Page 3 content"]
        chunks = processor.create_chunks(page_list, self.filename, self.file_type)
        
        assert len(chunks) == 3
        assert all(chunk['metadata']['chunk_method'] == 'page_based' for chunk in chunks)
        assert chunks[0]['content'] == "Page 1 content"
        assert chunks[1]['content'] == "Page 2 content"
        assert chunks[2]['content'] == "Page 3 content"
    
    def test_create_chunks_fallback_strategy(self):
        """Test create_chunks with unknown strategy falls back to sentence"""
        processor = DocumentProcessor(chunking_strategy='unknown')
        
        chunks = processor.create_chunks(self.sample_text, self.filename, self.file_type)
        
        assert len(chunks) > 0
        assert all(chunk['metadata']['chunk_method'] == 'sentence_based' for chunk in chunks)


class TestDocumentProcessorSentenceChunking:
    """Test sentence-based chunking functionality"""
    
    def test_chunk_by_sentences_basic(self):
        """Test basic sentence chunking"""
        processor = DocumentProcessor(max_sentences_per_chunk=2)
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = processor._chunk_by_sentences(content, "test.txt", "txt")
        
        assert len(chunks) >= 1
        assert all('content' in chunk for chunk in chunks)
        assert all(chunk['metadata']['chunk_method'] == 'sentence_based' for chunk in chunks)
        assert all(chunk['metadata']['max_sentences_per_chunk'] == 2 for chunk in chunks)
    
    def test_chunk_by_sentences_with_list_input(self):
        """Test sentence chunking with list input"""
        processor = DocumentProcessor(max_sentences_per_chunk=2)
        content = ["First line.", "Second line.", "Third line."]
        
        chunks = processor._chunk_by_sentences(content, "test.txt", "txt")
        
        assert len(chunks) >= 1
        assert all('content' in chunk for chunk in chunks)
    
    def test_chunk_by_sentences_with_split_newlines(self):
        """Test sentence chunking with split_newlines parameter"""
        processor = DocumentProcessor(max_sentences_per_chunk=5, split_newlines=2)
        content = "First sentence. Second sentence.\n\nThird sentence. Fourth sentence."
        
        chunks = processor._chunk_by_sentences(content, "test.txt", "txt")
        
        assert len(chunks) >= 1
        assert all(chunk['metadata']['split_newlines'] == 2 for chunk in chunks)


class TestDocumentProcessorSlidingWindow:
    """Test sliding window chunking functionality"""
    
    def test_chunk_by_sliding_window_basic(self):
        """Test basic sliding window chunking"""
        processor = DocumentProcessor(chunk_size=3, chunk_overlap=1)
        content = "one two three four five six seven eight"
        
        chunks = processor._chunk_by_sliding_window(content, "test.txt", "txt")
        
        assert len(chunks) > 1
        assert all(chunk['metadata']['chunk_method'] == 'sliding_window' for chunk in chunks)
        assert all(chunk['metadata']['chunk_size_words'] == 3 for chunk in chunks)
        assert all(chunk['metadata']['overlap_size_words'] == 1 for chunk in chunks)
        
        # Check overlap
        first_chunk_words = chunks[0]['content'].split()
        second_chunk_words = chunks[1]['content'].split()
        assert len(first_chunk_words) == 3
        assert len(second_chunk_words) <= 3
    
    def test_chunk_by_sliding_window_with_list_input(self):
        """Test sliding window chunking with list input"""
        processor = DocumentProcessor(chunk_size=2, chunk_overlap=1)
        content = ["line one", "line two", "line three"]
        
        chunks = processor._chunk_by_sliding_window(content, "test.txt", "txt")
        
        assert len(chunks) >= 1
        assert all('content' in chunk for chunk in chunks)
    
    def test_chunk_by_sliding_window_empty_content(self):
        """Test sliding window chunking with empty content"""
        processor = DocumentProcessor(chunk_size=3, chunk_overlap=1)
        content = ""
        
        chunks = processor._chunk_by_sliding_window(content, "test.txt", "txt")
        
        assert chunks == []
    
    def test_chunk_by_sliding_window_metadata(self):
        """Test sliding window chunking metadata"""
        processor = DocumentProcessor(chunk_size=2, chunk_overlap=1)
        content = "word1 word2 word3 word4"
        
        chunks = processor._chunk_by_sliding_window(content, "test.txt", "txt")
        
        assert len(chunks) >= 2
        assert chunks[0]['metadata']['start_word'] == 0
        assert chunks[0]['metadata']['end_word'] == 2
        assert chunks[1]['metadata']['start_word'] == 1  # overlap
        assert chunks[1]['metadata']['end_word'] == 3


class TestDocumentProcessorParagraphChunking:
    """Test paragraph-based chunking functionality"""
    
    def test_chunk_by_paragraphs_basic(self):
        """Test basic paragraph chunking"""
        processor = DocumentProcessor()
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        chunks = processor._chunk_by_paragraphs(content, "test.txt", "txt")
        
        assert len(chunks) == 3
        assert chunks[0]['content'] == "First paragraph."
        assert chunks[1]['content'] == "Second paragraph."
        assert chunks[2]['content'] == "Third paragraph."
        assert all(chunk['metadata']['chunk_method'] == 'paragraph_based' for chunk in chunks)
    
    def test_chunk_by_paragraphs_with_list_input(self):
        """Test paragraph chunking with list input"""
        processor = DocumentProcessor()
        content = ["First line", "Second line", "", "Third line"]
        
        chunks = processor._chunk_by_paragraphs(content, "test.txt", "txt")
        
        assert len(chunks) >= 1
        assert all('content' in chunk for chunk in chunks)
    
    def test_chunk_by_paragraphs_with_whitespace(self):
        """Test paragraph chunking with various whitespace"""
        processor = DocumentProcessor()
        content = "Para 1.\n  \n  Para 2.  \n\n\n  Para 3."
        
        chunks = processor._chunk_by_paragraphs(content, "test.txt", "txt")
        
        assert len(chunks) == 3
        assert chunks[0]['content'] == "Para 1."
        assert chunks[1]['content'] == "Para 2."
        assert chunks[2]['content'] == "Para 3."
    
    def test_chunk_by_paragraphs_empty_paragraphs(self):
        """Test paragraph chunking skips empty paragraphs"""
        processor = DocumentProcessor()
        content = "Para 1.\n\n\n\nPara 2.\n\n"
        
        chunks = processor._chunk_by_paragraphs(content, "test.txt", "txt")
        
        assert len(chunks) == 2
        assert chunks[0]['content'] == "Para 1."
        assert chunks[1]['content'] == "Para 2."


class TestDocumentProcessorPageChunking:
    """Test page-based chunking functionality"""
    
    def test_chunk_by_pages_with_list(self):
        """Test page chunking with list input (like PDF)"""
        processor = DocumentProcessor()
        content = ["Page 1 content", "Page 2 content", "", "Page 3 content"]
        
        chunks = processor._chunk_by_pages(content, "test.txt", "txt")
        
        assert len(chunks) == 3  # Empty page should be skipped
        assert chunks[0]['content'] == "Page 1 content"
        assert chunks[1]['content'] == "Page 2 content"
        assert chunks[2]['content'] == "Page 3 content"
        assert all(chunk['metadata']['chunk_method'] == 'page_based' for chunk in chunks)
    
    def test_chunk_by_pages_with_form_feeds(self):
        """Test page chunking with form feed characters"""
        processor = DocumentProcessor()
        content = "Page 1 content\fPage 2 content\fPage 3 content"
        
        chunks = processor._chunk_by_pages(content, "test.txt", "txt")
        
        assert len(chunks) == 3
        assert chunks[0]['content'] == "Page 1 content"
        assert chunks[1]['content'] == "Page 2 content"
        assert chunks[2]['content'] == "Page 3 content"
    
    def test_chunk_by_pages_with_page_markers(self):
        """Test page chunking with page break markers"""
        processor = DocumentProcessor()
        content = "Page 1 content---PAGE---Page 2 content---PAGE---Page 3 content"
        
        chunks = processor._chunk_by_pages(content, "test.txt", "txt")
        
        assert len(chunks) == 3
        assert chunks[0]['content'] == "Page 1 content"
        assert chunks[1]['content'] == "Page 2 content"
        assert chunks[2]['content'] == "Page 3 content"
    
    def test_chunk_by_pages_fallback_chunking(self):
        """Test page chunking fallback for plain text"""
        processor = DocumentProcessor()
        # Create content that will trigger fallback chunking
        words = ["word"] * 1000  # 1000 words
        content = " ".join(words)
        
        chunks = processor._chunk_by_pages(content, "test.txt", "txt")
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(chunk['metadata']['chunk_method'] == 'page_based' for chunk in chunks)
        assert all('page_number' in chunk['metadata'] for chunk in chunks)


class TestDocumentProcessorFileExtraction:
    """Test file extraction methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor()
    
    @patch('signalwire_agents.search.document_processor.magic', None)
    def test_extract_text_from_file_no_magic(self):
        """Test file extraction without magic library"""
        with patch.object(self.processor, '_extract_text') as mock_extract:
            mock_extract.return_value = "test content"
            
            result = self.processor._extract_text_from_file("test.txt")
            
            mock_extract.assert_called_once_with("test.txt")
            assert result == "test content"
    
    @patch('signalwire_agents.search.document_processor.magic')
    def test_extract_text_from_file_with_magic(self, mock_magic):
        """Test file extraction with magic library"""
        mock_mime = Mock()
        mock_mime.from_file.return_value = "text/plain"
        mock_magic.Magic.return_value = mock_mime
        
        with patch.object(self.processor, '_extract_text') as mock_extract:
            mock_extract.return_value = "test content"
            
            result = self.processor._extract_text_from_file("test.txt")
            
            mock_extract.assert_called_once_with("test.txt")
            assert result == "test content"
    
    def test_extract_text_from_file_pdf(self):
        """Test PDF file extraction"""
        with patch.object(self.processor, '_extract_pdf') as mock_extract:
            mock_extract.return_value = ["page 1", "page 2"]
            
            result = self.processor._extract_text_from_file("test.pdf")
            
            mock_extract.assert_called_once_with("test.pdf")
            assert result == ["page 1", "page 2"]
    
    def test_extract_text_from_file_docx(self):
        """Test DOCX file extraction"""
        with patch.object(self.processor, '_extract_docx') as mock_extract:
            mock_extract.return_value = ["paragraph 1", "paragraph 2"]
            
            result = self.processor._extract_text_from_file("test.docx")
            
            mock_extract.assert_called_once_with("test.docx")
            assert result == ["paragraph 1", "paragraph 2"]
    
    def test_extract_text_from_file_html(self):
        """Test HTML file extraction"""
        # HTML files with fallback detection go to _extract_text due to 'text' in 'text/html'
        with patch('signalwire_agents.search.document_processor.magic', None):
            with patch.object(self.processor, '_extract_text') as mock_extract:
                mock_extract.return_value = "html content"
                
                result = self.processor._extract_text_from_file("test.html")
                
                mock_extract.assert_called_once_with("test.html")
                assert result == "html content"
    
    def test_extract_text_from_file_markdown(self):
        """Test Markdown file extraction"""
        # Markdown files with fallback detection go to _extract_text due to 'text' in 'text/plain'
        with patch('signalwire_agents.search.document_processor.magic', None):
            with patch.object(self.processor, '_extract_text') as mock_extract:
                mock_extract.return_value = "markdown content"
                
                result = self.processor._extract_text_from_file("test.md")
                
                mock_extract.assert_called_once_with("test.md")
                assert result == "markdown content"
    
    def test_extract_text_from_file_unsupported(self):
        """Test unsupported file type"""
        with patch('signalwire_agents.search.document_processor.magic') as mock_magic:
            mock_mime = Mock()
            mock_mime.from_file.return_value = "application/unknown"
            mock_magic.Magic.return_value = mock_mime
            
            result = self.processor._extract_text_from_file("test.unknown")
            
            assert "Unsupported file type" in result


class TestDocumentProcessorSpecificExtractors:
    """Test specific file format extractors"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor()
    
    @patch('signalwire_agents.search.document_processor.pdfplumber', None)
    def test_extract_pdf_no_library(self):
        """Test PDF extraction without pdfplumber"""
        result = self.processor._extract_pdf("test.pdf")
        
        assert "pdfplumber not available" in result
    
    @patch('signalwire_agents.search.document_processor.pdfplumber')
    def test_extract_pdf_success(self, mock_pdfplumber):
        """Test successful PDF extraction"""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        result = self.processor._extract_pdf("test.pdf")
        
        assert result == ["Page 1 content", "Page 2 content"]
    
    @patch('signalwire_agents.search.document_processor.pdfplumber')
    def test_extract_pdf_error(self, mock_pdfplumber):
        """Test PDF extraction with error"""
        mock_pdfplumber.open.side_effect = Exception("PDF error")
        
        result = self.processor._extract_pdf("test.pdf")
        
        assert "Error processing PDF" in result
    
    @patch('signalwire_agents.search.document_processor.DocxDocument', None)
    def test_extract_docx_no_library(self):
        """Test DOCX extraction without python-docx"""
        result = self.processor._extract_docx("test.docx")
        
        assert "python-docx not available" in result
    
    @patch('signalwire_agents.search.document_processor.DocxDocument')
    def test_extract_docx_success(self, mock_docx):
        """Test successful DOCX extraction"""
        mock_para1 = Mock()
        mock_para1.text = "Paragraph 1"
        mock_para2 = Mock()
        mock_para2.text = "Paragraph 2"
        mock_para3 = Mock()
        mock_para3.text = ""  # Empty paragraph should be filtered
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_docx.return_value = mock_doc
        
        result = self.processor._extract_docx("test.docx")
        
        assert result == ["Paragraph 1", "Paragraph 2"]
    
    @patch('signalwire_agents.search.document_processor.DocxDocument')
    def test_extract_docx_error(self, mock_docx):
        """Test DOCX extraction with error"""
        mock_docx.side_effect = Exception("DOCX error")
        
        result = self.processor._extract_docx("test.docx")
        
        assert "Error processing DOCX" in result
    
    def test_extract_text_success(self):
        """Test successful text file extraction"""
        with patch('builtins.open', mock_open(read_data="test content")):
            result = self.processor._extract_text("test.txt")
            
            assert result == "test content"
    
    def test_extract_text_error(self):
        """Test text file extraction with error"""
        with patch('builtins.open', side_effect=Exception("File error")):
            result = self.processor._extract_text("test.txt")
            
            assert "Error processing TXT" in result
    
    @patch('signalwire_agents.search.document_processor.BeautifulSoup', None)
    def test_extract_html_no_library(self):
        """Test HTML extraction without BeautifulSoup"""
        result = self.processor._extract_html("test.html")
        
        assert "beautifulsoup4 not available" in result
    
    @patch('signalwire_agents.search.document_processor.BeautifulSoup')
    def test_extract_html_success(self, mock_bs):
        """Test successful HTML extraction"""
        mock_soup = Mock()
        mock_soup.get_text.return_value = "HTML content"
        mock_bs.return_value = mock_soup
        
        with patch('builtins.open', mock_open(read_data="<html><body>HTML content</body></html>")):
            result = self.processor._extract_html("test.html")
            
            assert result == "HTML content"
    
    @patch('signalwire_agents.search.document_processor.markdown', None)
    def test_extract_markdown_no_library(self):
        """Test Markdown extraction without markdown library"""
        with patch('builtins.open', mock_open(read_data="# Header\nContent")):
            result = self.processor._extract_markdown("test.md")
            
            # Should fallback to plain text
            assert result == "# Header\nContent"
    
    @patch('signalwire_agents.search.document_processor.markdown')
    def test_extract_markdown_success(self, mock_markdown):
        """Test successful Markdown extraction"""
        mock_markdown.markdown.return_value = "<h1>Header</h1><p>Content</p>"
        
        with patch('builtins.open', mock_open(read_data="# Header\nContent")):
            with patch('signalwire_agents.search.document_processor.BeautifulSoup') as mock_bs:
                mock_soup = Mock()
                mock_soup.get_text.return_value = "Header Content"
                mock_bs.return_value = mock_soup
                
                result = self.processor._extract_markdown("test.md")
                
                assert result == "Header Content"


class TestDocumentProcessorUtilities:
    """Test utility methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor()
    
    def test_create_chunk_basic(self):
        """Test basic chunk creation"""
        chunk = self.processor._create_chunk(
            content="Test content",
            filename="test.txt",
            section="Section 1"
        )
        
        assert chunk['content'] == "Test content"
        assert chunk['filename'] == "test.txt"
        assert chunk['section'] == "Section 1"
        assert 'metadata' in chunk
        assert chunk['metadata']['file_type'] == 'txt'
        assert chunk['metadata']['chunk_size'] == len("Test content")
        assert chunk['metadata']['word_count'] == 2
    
    def test_create_chunk_with_metadata(self):
        """Test chunk creation with custom metadata"""
        custom_metadata = {"custom_field": "custom_value"}
        
        chunk = self.processor._create_chunk(
            content="Test content",
            filename="test.txt",
            metadata=custom_metadata
        )
        
        assert chunk['metadata']['custom_field'] == "custom_value"
        assert chunk['metadata']['file_type'] == 'txt'  # Base metadata should still be there
    
    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_create_chunk_sentence_count_with_nltk(self, mock_sent_tokenize):
        """Test chunk creation with NLTK sentence tokenization"""
        mock_sent_tokenize.return_value = ["Sentence 1.", "Sentence 2."]
        
        chunk = self.processor._create_chunk(
            content="Sentence 1. Sentence 2.",
            filename="test.txt"
        )
        
        assert chunk['metadata']['sentence_count'] == 2
        mock_sent_tokenize.assert_called_once_with("Sentence 1. Sentence 2.")
    
    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_create_chunk_sentence_count_fallback(self):
        """Test chunk creation with fallback sentence counting"""
        chunk = self.processor._create_chunk(
            content="Sentence 1. Sentence 2. Sentence 3.",
            filename="test.txt"
        )
        
        assert chunk['metadata']['sentence_count'] == 3
    
    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_create_chunk_sentence_count_error(self, mock_sent_tokenize):
        """Test chunk creation with sentence counting error"""
        mock_sent_tokenize.side_effect = Exception("NLTK error")
        
        chunk = self.processor._create_chunk(
            content="Sentence 1. Sentence 2.",
            filename="test.txt"
        )
        
        # Should fallback to period counting
        assert chunk['metadata']['sentence_count'] == 2
    
    def test_find_best_split_point_with_empty_lines(self):
        """Test finding best split point with paragraph boundaries"""
        lines = ["line1", "line2", "", "line4", "line5", "line6"]
        
        split_point = self.processor._find_best_split_point(lines)
        
        # The algorithm searches backwards from the end in the last 25% of the chunk
        # With 6 lines, it starts searching from line 4 (3//4 * 6 = 4) backwards
        # It should find the empty line at index 2, but since it's outside the search range,
        # it will return the 75% point which is max(1, 6 * 3 // 4) = 4
        assert split_point == 4
    
    def test_find_best_split_point_no_empty_lines(self):
        """Test finding best split point without paragraph boundaries"""
        lines = ["line1", "line2", "line3", "line4", "line5", "line6"]
        
        split_point = self.processor._find_best_split_point(lines)
        
        # Should split at 75% of chunk size
        expected = max(1, len(lines) * 3 // 4)
        assert split_point == expected
    
    def test_get_overlap_lines_basic(self):
        """Test getting overlap lines"""
        processor = DocumentProcessor(chunk_overlap=50)  # 50 characters - large enough to capture lines
        lines = ["short", "medium line", "longer line here"]
        
        overlap = processor._get_overlap_lines(lines)
        
        # Should include lines that fit within overlap size
        assert len(overlap) > 0
        assert all(isinstance(line, str) for line in overlap)
    
    def test_get_overlap_lines_empty(self):
        """Test getting overlap lines with empty input"""
        overlap = self.processor._get_overlap_lines([])
        
        assert overlap == []


class TestDocumentProcessorEdgeCases:
    """Test edge cases and error handling"""
    
    def test_chunking_with_empty_content(self):
        """Test chunking with empty content"""
        processor = DocumentProcessor()
        
        chunks = processor.create_chunks("", "test.txt", "txt")
        
        # Should handle empty content gracefully
        assert isinstance(chunks, list)
    
    def test_chunking_with_whitespace_only(self):
        """Test chunking with whitespace-only content"""
        processor = DocumentProcessor()
        
        chunks = processor.create_chunks("   \n\n   ", "test.txt", "txt")
        
        # Should handle whitespace-only content gracefully
        assert isinstance(chunks, list)
    
    def test_sliding_window_with_small_content(self):
        """Test sliding window with content smaller than chunk size"""
        processor = DocumentProcessor(chunking_strategy='sliding', chunk_size=10, chunk_overlap=2)
        
        chunks = processor.create_chunks("small", "test.txt", "txt")
        
        assert len(chunks) == 1
        assert chunks[0]['content'] == "small"
    
    def test_paragraph_chunking_no_paragraphs(self):
        """Test paragraph chunking with no paragraph breaks"""
        processor = DocumentProcessor(chunking_strategy='paragraph')
        
        chunks = processor.create_chunks("Single line of text", "test.txt", "txt")
        
        assert len(chunks) == 1
        assert chunks[0]['content'] == "Single line of text"
    
    def test_page_chunking_empty_pages(self):
        """Test page chunking with empty pages"""
        processor = DocumentProcessor(chunking_strategy='page')
        
        chunks = processor.create_chunks(["", "  ", "Content", ""], "test.txt", "txt")
        
        assert len(chunks) == 1
        assert chunks[0]['content'] == "Content" 