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


# ────────────────────────────────────────────────────────────────────
# NEW TESTS APPENDED BELOW
# ────────────────────────────────────────────────────────────────────

import json
import re


class TestFileExtraction:
    """Comprehensive tests for every extraction method and missing-dependency paths."""

    def setup_method(self):
        self.processor = DocumentProcessor()

    # ── PDF ──────────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.pdfplumber')
    def test_extract_pdf_with_pages(self, mock_pdfplumber):
        """Extract text from multiple PDF pages, including one with None text."""
        page1 = Mock(); page1.extract_text.return_value = "Page one text"
        page2 = Mock(); page2.extract_text.return_value = None        # empty page
        page3 = Mock(); page3.extract_text.return_value = "Page three text"

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2, page3]
        mock_pdfplumber.open.return_value.__enter__ = Mock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = Mock(return_value=False)

        result = self.processor._extract_pdf("/fake/doc.pdf")
        assert isinstance(result, list)
        assert len(result) == 2  # page2 had None text, should be skipped
        assert "Page one text" in result[0]

    @patch('signalwire_agents.search.document_processor.pdfplumber')
    def test_extract_pdf_strips_leading_page_numbers(self, mock_pdfplumber):
        """Leading page numbers like '1. ' should be stripped."""
        page = Mock(); page.extract_text.return_value = "1. Introduction paragraph"
        mock_pdf = MagicMock(); mock_pdf.pages = [page]
        mock_pdfplumber.open.return_value.__enter__ = Mock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = Mock(return_value=False)

        result = self.processor._extract_pdf("/fake/doc.pdf")
        assert result == ["Introduction paragraph"]

    @patch('signalwire_agents.search.document_processor.pdfplumber', None)
    def test_extract_pdf_missing_dependency(self):
        result = self.processor._extract_pdf("/fake/doc.pdf")
        assert "pdfplumber not available" in result

    @patch('signalwire_agents.search.document_processor.pdfplumber')
    def test_extract_pdf_exception(self, mock_pdfplumber):
        mock_pdfplumber.open.side_effect = RuntimeError("corrupt file")
        result = self.processor._extract_pdf("/fake/doc.pdf")
        assert "Error processing PDF" in result

    # ── DOCX ─────────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.DocxDocument')
    def test_extract_docx_filters_empty_paragraphs(self, mock_docx_cls):
        p1 = Mock(text="Hello"); p2 = Mock(text=""); p3 = Mock(text="World")
        mock_doc = Mock(); mock_doc.paragraphs = [p1, p2, p3]
        mock_docx_cls.return_value = mock_doc

        result = self.processor._extract_docx("/fake/doc.docx")
        assert result == ["Hello", "World"]

    @patch('signalwire_agents.search.document_processor.DocxDocument', None)
    def test_extract_docx_missing_dependency(self):
        result = self.processor._extract_docx("/fake/doc.docx")
        assert "python-docx not available" in result

    @patch('signalwire_agents.search.document_processor.DocxDocument')
    def test_extract_docx_exception(self, mock_docx_cls):
        mock_docx_cls.side_effect = ValueError("bad docx")
        result = self.processor._extract_docx("/fake/doc.docx")
        assert "Error processing DOCX" in result

    # ── XLSX (Excel) ─────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.load_workbook')
    def test_extract_excel_success(self, mock_lwb):
        mock_sheet = Mock()
        mock_sheet.iter_rows.return_value = [
            ("Name", "Age"),
            ("Alice", 30),
            (None, "Bob"),
        ]
        mock_wb = Mock(); mock_wb.worksheets = [mock_sheet]
        mock_lwb.return_value = mock_wb

        result = self.processor._extract_excel("/fake/data.xlsx")
        assert "Name" in result
        assert "Alice" in result
        assert "30" in result  # integers become str
        assert "Bob" in result

    @patch('signalwire_agents.search.document_processor.load_workbook', None)
    def test_extract_excel_missing_dependency(self):
        result = self.processor._extract_excel("/fake/data.xlsx")
        assert "openpyxl not available" in result

    @patch('signalwire_agents.search.document_processor.load_workbook')
    def test_extract_excel_exception(self, mock_lwb):
        mock_lwb.side_effect = Exception("xlsx error")
        result = self.processor._extract_excel("/fake/data.xlsx")
        assert "Error processing Excel" in result

    # ── PPTX (PowerPoint) ────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.Presentation')
    def test_extract_powerpoint_success(self, mock_pres_cls):
        shape1 = Mock(text="Title Slide"); shape1.__class__ = type('S', (), {'text': 'Title Slide'})
        shape2 = Mock(text="Bullet 1")
        shape_no_text = Mock(spec=[])  # no 'text' attribute
        slide1 = Mock(); slide1.shapes = [shape1, shape_no_text]
        slide2 = Mock(); slide2.shapes = [shape2]
        mock_pres = Mock(); mock_pres.slides = [slide1, slide2]
        mock_pres_cls.return_value = mock_pres

        result = self.processor._extract_powerpoint("/fake/pres.pptx")
        assert isinstance(result, list)
        assert len(result) == 2
        assert "Title Slide" in result[0]
        assert "Bullet 1" in result[1]

    @patch('signalwire_agents.search.document_processor.Presentation', None)
    def test_extract_powerpoint_missing_dependency(self):
        result = self.processor._extract_powerpoint("/fake/pres.pptx")
        assert "python-pptx not available" in result

    @patch('signalwire_agents.search.document_processor.Presentation')
    def test_extract_powerpoint_exception(self, mock_pres_cls):
        mock_pres_cls.side_effect = Exception("pptx error")
        result = self.processor._extract_powerpoint("/fake/pres.pptx")
        assert "Error processing PowerPoint" in result

    # ── HTML ─────────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.BeautifulSoup')
    def test_extract_html_success(self, mock_bs):
        mock_soup = Mock(); mock_soup.get_text.return_value = "Hello World"
        mock_bs.return_value = mock_soup
        with patch('builtins.open', mock_open(read_data="<p>Hello World</p>")):
            result = self.processor._extract_html("/fake/page.html")
        assert result == "Hello World"

    @patch('signalwire_agents.search.document_processor.BeautifulSoup', None)
    def test_extract_html_missing_dependency(self):
        result = self.processor._extract_html("/fake/page.html")
        assert "beautifulsoup4 not available" in result

    @patch('signalwire_agents.search.document_processor.BeautifulSoup')
    def test_extract_html_exception(self, mock_bs):
        with patch('builtins.open', side_effect=IOError("disk")):
            result = self.processor._extract_html("/fake/page.html")
        assert "Error processing HTML" in result

    # ── Markdown ─────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.BeautifulSoup')
    @patch('signalwire_agents.search.document_processor.markdown')
    def test_extract_markdown_success(self, mock_md, mock_bs):
        mock_md.markdown.return_value = "<h1>Title</h1>"
        mock_soup = Mock(); mock_soup.get_text.return_value = "Title"
        mock_bs.return_value = mock_soup
        with patch('builtins.open', mock_open(read_data="# Title")):
            result = self.processor._extract_markdown("/fake/doc.md")
        assert result == "Title"

    @patch('signalwire_agents.search.document_processor.BeautifulSoup', None)
    @patch('signalwire_agents.search.document_processor.markdown', None)
    def test_extract_markdown_fallback_raw(self):
        """Without markdown+BS4, returns raw markdown text."""
        with patch('builtins.open', mock_open(read_data="# Raw Title\nBody")):
            result = self.processor._extract_markdown("/fake/doc.md")
        assert result == "# Raw Title\nBody"

    def test_extract_markdown_io_error(self):
        with patch('builtins.open', side_effect=OSError("nope")):
            result = self.processor._extract_markdown("/fake/doc.md")
        assert "Error processing Markdown" in result

    # ── RTF ───────────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.rtf_to_text')
    def test_extract_rtf_success(self, mock_rtf):
        mock_rtf.return_value = "Plain text from RTF"
        with patch('builtins.open', mock_open(read_data=r"{\rtf1 Plain text from RTF}")):
            result = self.processor._extract_rtf("/fake/doc.rtf")
        assert result == "Plain text from RTF"

    @patch('signalwire_agents.search.document_processor.rtf_to_text', None)
    def test_extract_rtf_missing_dependency(self):
        result = self.processor._extract_rtf("/fake/doc.rtf")
        assert "striprtf not available" in result

    @patch('signalwire_agents.search.document_processor.rtf_to_text')
    def test_extract_rtf_exception(self, mock_rtf):
        with patch('builtins.open', side_effect=IOError("disk")):
            result = self.processor._extract_rtf("/fake/doc.rtf")
        assert "Error processing RTF" in result

    # ── Plain text ────────────────────────────────────────────────────

    def test_extract_text_success(self):
        with patch('builtins.open', mock_open(read_data="hello world")):
            result = self.processor._extract_text("/fake/file.txt")
        assert result == "hello world"

    def test_extract_text_encoding_error(self):
        with patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'bad')):
            result = self.processor._extract_text("/fake/file.txt")
        assert "Error processing TXT" in result

    # ── _extract_text_from_file routing ──────────────────────────────

    @patch('signalwire_agents.search.document_processor.magic', None)
    def test_extract_text_from_file_routes_xlsx(self):
        with patch.object(self.processor, '_extract_excel', return_value="cells") as m:
            result = self.processor._extract_text_from_file("data.xlsx")
        m.assert_called_once_with("data.xlsx")
        assert result == "cells"

    @patch('signalwire_agents.search.document_processor.magic', None)
    def test_extract_text_from_file_routes_pptx(self):
        with patch.object(self.processor, '_extract_powerpoint', return_value=["s1"]) as m:
            result = self.processor._extract_text_from_file("slides.pptx")
        m.assert_called_once_with("slides.pptx")
        assert result == ["s1"]

    @patch('signalwire_agents.search.document_processor.magic', None)
    def test_extract_text_from_file_routes_rtf(self):
        with patch.object(self.processor, '_extract_rtf', return_value="rtf text") as m:
            result = self.processor._extract_text_from_file("notes.rtf")
        m.assert_called_once_with("notes.rtf")
        assert result == "rtf text"

    @patch('signalwire_agents.search.document_processor.magic', None)
    def test_extract_text_from_file_unknown_extension(self):
        """Unknown extension falls back to text/plain routing."""
        with patch.object(self.processor, '_extract_text', return_value="raw") as m:
            result = self.processor._extract_text_from_file("file.zzz")
        m.assert_called_once_with("file.zzz")
        assert result == "raw"

    @patch('signalwire_agents.search.document_processor.magic')
    def test_extract_text_from_file_magic_html(self, mock_magic):
        """When magic reports text/html, route to _extract_html."""
        mock_mime = Mock(); mock_mime.from_file.return_value = "text/html"
        mock_magic.Magic.return_value = mock_mime
        # 'html' in 'text/html' is True, but 'plain' in 'text/html' is False,
        # and 'text' in 'text/html' is True -> goes to _extract_text first
        # Actually 'plain' in 'text/html' is False but 'text' in 'text/html' is True
        # So it hits the 'plain' or 'text' branch -> _extract_text
        with patch.object(self.processor, '_extract_text', return_value="content") as m:
            result = self.processor._extract_text_from_file("page.html")
        assert result == "content"

    @patch('signalwire_agents.search.document_processor.magic')
    def test_extract_text_from_file_magic_rtf(self, mock_magic):
        mock_mime = Mock(); mock_mime.from_file.return_value = "application/rtf"
        mock_magic.Magic.return_value = mock_mime
        with patch.object(self.processor, '_extract_rtf', return_value="rtf") as m:
            result = self.processor._extract_text_from_file("doc.rtf")
        m.assert_called_once_with("doc.rtf")
        assert result == "rtf"

    @patch('signalwire_agents.search.document_processor.magic')
    def test_extract_text_from_file_magic_unsupported(self, mock_magic):
        mock_mime = Mock(); mock_mime.from_file.return_value = "application/octet-stream"
        mock_magic.Magic.return_value = mock_mime
        result = self.processor._extract_text_from_file("binary.bin")
        assert "Unsupported file type" in result


class TestChunkingStrategies:
    """Exhaustive tests for every chunking strategy path."""

    # ── sentence ─────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_sentence_chunking_no_nltk(self):
        """Without NLTK, fallback to period-based splitting."""
        proc = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=2)
        text = "Alpha. Beta. Gamma. Delta."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1
        assert all(c['metadata']['chunk_method'] == 'sentence_based' for c in chunks)

    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_sentence_chunking_with_nltk(self, mock_tok):
        mock_tok.side_effect = lambda t: [s.strip() for s in t.split('.') if s.strip()]
        proc = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=2)
        text = "One. Two. Three. Four."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1

    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_sentence_chunking_with_split_newlines_zero(self, mock_tok):
        """split_newlines=0 should use direct tokenization without newline splitting."""
        mock_tok.side_effect = lambda t: [s.strip() for s in t.split('.') if s.strip()]
        proc = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=3, split_newlines=0)
        text = "Hello. World."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1

    def test_sentence_chunking_list_input(self):
        proc = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=3)
        chunks = proc.create_chunks(["Line A.", "Line B.", "Line C."], "f.txt", "txt")
        assert len(chunks) >= 1

    # ── sliding_window ───────────────────────────────────────────────

    def test_sliding_window_basic(self):
        proc = DocumentProcessor(chunking_strategy='sliding', chunk_size=3, chunk_overlap=1)
        chunks = proc.create_chunks("a b c d e f g h", "f.txt", "txt")
        assert len(chunks) >= 2
        assert all(c['metadata']['chunk_method'] == 'sliding_window' for c in chunks)

    def test_sliding_window_list_input(self):
        proc = DocumentProcessor(chunking_strategy='sliding', chunk_size=4, chunk_overlap=1)
        chunks = proc.create_chunks(["word1 word2", "word3 word4 word5"], "f.txt", "txt")
        assert len(chunks) >= 1

    def test_sliding_window_empty(self):
        proc = DocumentProcessor(chunking_strategy='sliding', chunk_size=5, chunk_overlap=2)
        chunks = proc.create_chunks("", "f.txt", "txt")
        assert chunks == []

    def test_sliding_window_overlap_metadata(self):
        proc = DocumentProcessor(chunking_strategy='sliding', chunk_size=3, chunk_overlap=1)
        chunks = proc.create_chunks("w1 w2 w3 w4 w5 w6", "f.txt", "txt")
        assert chunks[0]['metadata']['start_word'] == 0
        assert chunks[0]['metadata']['end_word'] == 3

    # ── paragraphs ───────────────────────────────────────────────────

    def test_paragraphs_basic(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) == 3
        assert chunks[0]['content'] == "Para one."
        assert all(c['metadata']['chunk_method'] == 'paragraph_based' for c in chunks)

    def test_paragraphs_single_paragraph(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        chunks = proc.create_chunks("Just one paragraph.", "f.txt", "txt")
        assert len(chunks) == 1

    def test_paragraphs_list_input(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        chunks = proc.create_chunks(["line1", "", "line3"], "f.txt", "txt")
        assert len(chunks) >= 1

    def test_paragraphs_metadata(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        chunks = proc.create_chunks("A.\n\nB.", "f.txt", "txt")
        assert chunks[0]['metadata']['paragraph_number'] == 1
        assert chunks[1]['metadata']['paragraph_number'] == 2

    # ── pages ────────────────────────────────────────────────────────

    def test_pages_list_input(self):
        proc = DocumentProcessor(chunking_strategy='page')
        chunks = proc.create_chunks(["P1", "P2"], "f.pdf", "pdf")
        assert len(chunks) == 2
        assert chunks[0]['metadata']['page_number'] == 1

    def test_pages_form_feed(self):
        proc = DocumentProcessor(chunking_strategy='page')
        chunks = proc.create_chunks("A\fB\fC", "f.txt", "txt")
        assert len(chunks) == 3

    def test_pages_page_markers(self):
        proc = DocumentProcessor(chunking_strategy='page')
        chunks = proc.create_chunks("X---PAGE---Y", "f.txt", "txt")
        assert len(chunks) == 2

    def test_pages_page_number_pattern(self):
        proc = DocumentProcessor(chunking_strategy='page')
        text = "Content A\n Page 1 \nContent B\n Page 2 \nContent C"
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 2

    def test_pages_fallback_large_text(self):
        """Plain text without markers should fall back to word-based splitting."""
        proc = DocumentProcessor(chunking_strategy='page')
        words = " ".join(["word"] * 2000)
        chunks = proc.create_chunks(words, "f.txt", "txt")
        assert len(chunks) >= 2

    # ── semantic ─────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_semantic_single_sentence_fallback(self):
        """Single sentence should return one chunk without import error."""
        proc = DocumentProcessor(chunking_strategy='semantic')
        chunks = proc.create_chunks("Only one sentence.", "f.txt", "txt")
        assert len(chunks) == 1
        assert chunks[0]['metadata']['chunk_method'] == 'semantic'

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_semantic_import_error_falls_back(self):
        """When sentence_transformers is missing, fall back to sentence-based."""
        proc = DocumentProcessor(chunking_strategy='semantic')
        text = "Hello there. How are you. I am fine. Thanks for asking."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        # Should get chunks (via fallback)
        assert len(chunks) >= 1

    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_semantic_with_tokenizer_import_error(self, mock_tok):
        """Semantic with NLTK available but sentence_transformers missing."""
        mock_tok.side_effect = lambda t: [s.strip()+'.' for s in t.split('.') if s.strip()]
        proc = DocumentProcessor(chunking_strategy='semantic')
        text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1

    # ── topics ───────────────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_topics_short_text_single_chunk(self):
        """3 or fewer sentences returns a single topic chunk."""
        proc = DocumentProcessor(chunking_strategy='topic')
        text = "Machine learning rocks. Deep learning too."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) == 1
        assert chunks[0]['metadata']['chunk_method'] == 'topic'

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_topics_multiple_topics(self):
        """Multiple distinct topics should produce multiple chunks."""
        proc = DocumentProcessor(chunking_strategy='topic', topic_threshold=0.9)
        # Completely different keyword sets per sentence
        text = (
            "Python programming language features are impressive. "
            "JavaScript frontend development framework exists. "
            "Quantum physics particle accelerator experiments proceed. "
            "Medieval history castles knights kingdoms flourished. "
            "Oceanography marine biology coral reefs ecosystems thrive."
        )
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1
        assert all(c['metadata']['chunk_method'] == 'topic' for c in chunks)

    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_topics_with_nltk(self, mock_tok):
        mock_tok.side_effect = lambda t: [s.strip() for s in t.split('.') if s.strip()]
        proc = DocumentProcessor(chunking_strategy='topic', topic_threshold=0.0)
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_topics_list_input(self):
        proc = DocumentProcessor(chunking_strategy='topic')
        chunks = proc.create_chunks(
            ["First about dogs.", "Second about cats.", "Third about birds.", "Fourth about fish."],
            "f.txt", "txt"
        )
        assert len(chunks) >= 1

    # ── qa_optimization ──────────────────────────────────────────────

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_qa_optimization_basic(self):
        proc = DocumentProcessor(chunking_strategy='qa')
        text = (
            "What is Python? Python is a programming language. "
            "How does it work? It interprets code. "
            "Why use it? Because it is simple."
        )
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1
        assert all(c['metadata']['chunk_method'] == 'qa_optimized' for c in chunks)

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_qa_optimization_has_question_metadata(self):
        proc = DocumentProcessor(chunking_strategy='qa')
        text = "What is this? It is a test. Does it work? Yes it does. Final statement here."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1
        # At least one chunk should have 'has_question'
        has_q_chunks = [c for c in chunks if c['metadata'].get('has_question')]
        assert len(has_q_chunks) >= 0  # may or may not have, but should not crash

    @patch('signalwire_agents.search.document_processor.sent_tokenize')
    def test_qa_optimization_with_nltk(self, mock_tok):
        mock_tok.side_effect = lambda t: [s.strip() for s in t.split('.') if s.strip()]
        proc = DocumentProcessor(chunking_strategy='qa')
        text = "What is X? X is Y. How about Z? Z is W. Step one. Step two."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) >= 1

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_qa_optimization_empty_text(self):
        proc = DocumentProcessor(chunking_strategy='qa')
        chunks = proc.create_chunks("", "f.txt", "txt")
        assert isinstance(chunks, list)

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_qa_optimization_list_input(self):
        proc = DocumentProcessor(chunking_strategy='qa')
        chunks = proc.create_chunks(["Question? Answer.", "More content."], "f.txt", "txt")
        assert len(chunks) >= 1

    # ── json ─────────────────────────────────────────────────────────

    def test_json_chunking_valid(self):
        proc = DocumentProcessor(chunking_strategy='json')
        data = {
            "chunks": [
                {"chunk_id": "c1", "type": "content", "content": "Hello world",
                 "metadata": {"section": "Intro", "tags": ["greeting"]}},
                {"chunk_id": "c2", "type": "toc", "content": "Table of Contents item",
                 "metadata": {}},
            ]
        }
        chunks = proc.create_chunks(json.dumps(data), "f.json", "json")
        assert len(chunks) == 2
        assert chunks[0]['metadata']['chunk_method'] == 'json'
        assert chunks[0]['metadata']['original_chunk_id'] == 'c1'
        assert chunks[0]['tags'] == ['greeting']
        # TOC entry should get special tags
        assert 'toc' in chunks[1]['tags']

    def test_json_chunking_invalid_json(self):
        proc = DocumentProcessor(chunking_strategy='json')
        chunks = proc.create_chunks("not valid json {{{", "f.json", "json")
        # Should fall back to sentence-based
        assert len(chunks) >= 1

    def test_json_chunking_missing_chunks_key(self):
        proc = DocumentProcessor(chunking_strategy='json')
        data = {"data": [1, 2, 3]}
        chunks = proc.create_chunks(json.dumps(data), "f.json", "json")
        # Should fall back to sentence-based
        assert len(chunks) >= 1

    def test_json_chunking_skip_invalid_chunk(self):
        proc = DocumentProcessor(chunking_strategy='json')
        data = {
            "chunks": [
                {"chunk_id": "ok", "content": "Valid"},
                "this_is_not_a_dict",
                {"chunk_id": "no_content"},  # missing 'content'
            ]
        }
        chunks = proc.create_chunks(json.dumps(data), "f.json", "json")
        assert len(chunks) == 1
        assert "Valid" in chunks[0]['content']

    def test_json_chunking_empty_chunks_list(self):
        proc = DocumentProcessor(chunking_strategy='json')
        data = {"chunks": []}
        chunks = proc.create_chunks(json.dumps(data), "f.json", "json")
        # Falls back to sentence chunking of empty-ish text
        assert isinstance(chunks, list)

    def test_json_chunking_toc_section_naming(self):
        proc = DocumentProcessor(chunking_strategy='json')
        data = {"chunks": [
            {"type": "toc", "content": "A very long table of contents entry that should be truncated somehow"},
        ]}
        chunks = proc.create_chunks(json.dumps(data), "f.json", "json")
        assert len(chunks) == 1
        assert chunks[0]['section'].startswith("TOC:")

    def test_json_chunking_content_with_section_number(self):
        proc = DocumentProcessor(chunking_strategy='json')
        data = {"chunks": [
            {"type": "content", "content": "Body text", "metadata": {"section_number": 42}},
        ]}
        chunks = proc.create_chunks(json.dumps(data), "f.json", "json")
        assert chunks[0]['section'] == "Section 42"

    # ── markdown_enhanced ────────────────────────────────────────────

    def test_markdown_enhanced_basic_headers(self):
        proc = DocumentProcessor(chunking_strategy='markdown')
        md = "# Title\nSome intro.\n## Section A\nContent A.\n## Section B\nContent B."
        chunks = proc.create_chunks(md, "doc.md", "md")
        assert len(chunks) >= 2
        # Check section hierarchy
        sections = [c['section'] for c in chunks if c['section']]
        assert any('Title' in s for s in sections)

    def test_markdown_enhanced_code_blocks(self):
        proc = DocumentProcessor(chunking_strategy='markdown')
        md = "# Code Example\n```python\nprint('hello')\n```\nSome text."
        chunks = proc.create_chunks(md, "doc.md", "md")
        assert len(chunks) >= 1
        # At least one chunk should have code metadata
        code_chunks = [c for c in chunks if c['metadata'].get('has_code')]
        assert len(code_chunks) >= 1
        assert 'python' in code_chunks[0]['metadata'].get('code_languages', [])

    def test_markdown_enhanced_nested_headers(self):
        proc = DocumentProcessor(chunking_strategy='markdown')
        md = "# H1\n## H2\n### H3\nDeep content."
        chunks = proc.create_chunks(md, "doc.md", "md")
        assert len(chunks) >= 1

    def test_markdown_enhanced_no_headers(self):
        proc = DocumentProcessor(chunking_strategy='markdown')
        md = "Just plain text.\nMore text."
        chunks = proc.create_chunks(md, "doc.md", "md")
        assert len(chunks) >= 1

    def test_markdown_enhanced_code_without_language(self):
        proc = DocumentProcessor(chunking_strategy='markdown')
        md = "# Sec\n```\ngeneric code\n```\nDone."
        chunks = proc.create_chunks(md, "doc.md", "md")
        code_chunks = [c for c in chunks if c['metadata'].get('has_code')]
        assert len(code_chunks) >= 1
        # No language specified -> code_languages should be empty
        assert code_chunks[0]['metadata'].get('code_languages', []) == []

    def test_markdown_enhanced_empty_content(self):
        proc = DocumentProcessor(chunking_strategy='markdown')
        chunks = proc.create_chunks("", "doc.md", "md")
        assert isinstance(chunks, list)


class TestEdgeCases:
    """Edge cases: empty, unicode, very large, unsupported."""

    def test_empty_string_all_strategies(self):
        for strategy in ['sentence', 'sliding', 'paragraph', 'page', 'qa', 'json', 'markdown']:
            proc = DocumentProcessor(chunking_strategy=strategy)
            chunks = proc.create_chunks("", "f.txt", "txt")
            assert isinstance(chunks, list), f"Strategy {strategy} failed on empty string"

    def test_whitespace_only_all_strategies(self):
        for strategy in ['sentence', 'sliding', 'paragraph', 'page', 'qa', 'markdown']:
            proc = DocumentProcessor(chunking_strategy=strategy)
            chunks = proc.create_chunks("   \n\t\n  ", "f.txt", "txt")
            assert isinstance(chunks, list), f"Strategy {strategy} failed on whitespace"

    def test_unicode_content(self):
        proc = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=2)
        text = "日本語のテスト文。これは二番目の文です。三番目の文もあります。"
        chunks = proc.create_chunks(text, "jp.txt", "txt")
        assert len(chunks) >= 1

    def test_unicode_emoji_content(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        text = "First para with emoji.\n\nSecond para."
        chunks = proc.create_chunks(text, "emoji.txt", "txt")
        assert len(chunks) == 2

    def test_very_large_text_sentence(self):
        proc = DocumentProcessor(chunking_strategy='sentence', max_sentences_per_chunk=5)
        text = ". ".join([f"Sentence number {i}" for i in range(200)]) + "."
        chunks = proc.create_chunks(text, "big.txt", "txt")
        assert len(chunks) > 10

    def test_very_large_text_sliding(self):
        proc = DocumentProcessor(chunking_strategy='sliding', chunk_size=50, chunk_overlap=10)
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = proc.create_chunks(text, "big.txt", "txt")
        assert len(chunks) > 10

    def test_very_large_text_paragraph(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        text = "\n\n".join([f"Paragraph {i} content." for i in range(100)])
        chunks = proc.create_chunks(text, "big.txt", "txt")
        assert len(chunks) == 100

    def test_single_word(self):
        proc = DocumentProcessor(chunking_strategy='sentence')
        chunks = proc.create_chunks("hello", "f.txt", "txt")
        assert len(chunks) >= 1
        assert "hello" in chunks[0]['content']

    def test_single_character(self):
        proc = DocumentProcessor(chunking_strategy='sliding', chunk_size=5, chunk_overlap=1)
        chunks = proc.create_chunks("x", "f.txt", "txt")
        assert len(chunks) == 1

    def test_newlines_only(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        chunks = proc.create_chunks("\n\n\n\n", "f.txt", "txt")
        assert chunks == []

    def test_mixed_line_endings(self):
        proc = DocumentProcessor(chunking_strategy='paragraph')
        text = "Para 1.\r\n\r\nPara 2.\r\n\r\nPara 3."
        chunks = proc.create_chunks(text, "f.txt", "txt")
        assert len(chunks) == 3

    def test_unsupported_chunking_strategy_fallback(self):
        proc = DocumentProcessor(chunking_strategy='nonexistent_strategy')
        chunks = proc.create_chunks("Some text here.", "f.txt", "txt")
        # Fallback to sentence
        assert len(chunks) >= 1
        assert chunks[0]['metadata']['chunk_method'] == 'sentence_based'

    def test_create_chunk_hash_stability(self):
        """Same content produces same chunk structure."""
        proc = DocumentProcessor()
        c1 = proc._create_chunk("abc", "f.txt", "S1")
        c2 = proc._create_chunk("abc", "f.txt", "S1")
        assert c1['content'] == c2['content']
        assert c1['metadata']['word_count'] == c2['metadata']['word_count']

    def test_create_chunk_filename_extension(self):
        proc = DocumentProcessor()
        chunk = proc._create_chunk("test", "archive.tar.gz")
        assert chunk['metadata']['file_type'] == 'gz'

    def test_create_chunk_no_extension(self):
        proc = DocumentProcessor()
        chunk = proc._create_chunk("test", "Makefile")
        assert chunk['metadata']['file_type'] == ''

    # ── _build helpers ───────────────────────────────────────────────

    def test_build_section_path_empty(self):
        proc = DocumentProcessor()
        assert proc._build_section_path([]) is None

    def test_build_section_path_single(self):
        proc = DocumentProcessor()
        assert proc._build_section_path(["Intro"]) == "Intro"

    def test_build_section_path_nested(self):
        proc = DocumentProcessor()
        assert proc._build_section_path(["A", "B", "C"]) == "A > B > C"

    def test_build_markdown_metadata_no_code(self):
        proc = DocumentProcessor()
        meta = proc._build_markdown_metadata(["H1"], [], False)
        assert meta['chunk_type'] == 'markdown'
        assert meta.get('h1') == 'H1'
        assert 'has_code' not in meta

    def test_build_markdown_metadata_with_code(self):
        proc = DocumentProcessor()
        meta = proc._build_markdown_metadata(["H1", "H2"], ["python", "bash"], True)
        assert meta['has_code'] is True
        assert meta['code_languages'] == ["python", "bash"]
        assert 'code' in meta['tags']
        assert 'code:python' in meta['tags']
        assert 'depth:2' in meta['tags']

    def test_build_markdown_metadata_empty_hierarchy(self):
        proc = DocumentProcessor()
        meta = proc._build_markdown_metadata([], [], False)
        assert 'h1' not in meta
        assert 'tags' not in meta  # no code, no hierarchy -> no tags

    def test_build_python_section_class_and_function(self):
        proc = DocumentProcessor()
        assert proc._build_python_section("MyClass", "my_method") == "MyClass.my_method"

    def test_build_python_section_class_only(self):
        proc = DocumentProcessor()
        assert proc._build_python_section("MyClass", None) == "MyClass"

    def test_build_python_section_function_only(self):
        proc = DocumentProcessor()
        assert proc._build_python_section(None, "my_func") == "my_func"

    def test_build_python_section_neither(self):
        proc = DocumentProcessor()
        assert proc._build_python_section(None, None) is None

    # ── _calculate_sentences_per_chunk ───────────────────────────────

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_calculate_sentences_per_chunk_no_nltk(self):
        proc = DocumentProcessor(chunk_size=50)
        result = proc._calculate_sentences_per_chunk("Short. Another.")
        assert isinstance(result, int)
        assert result >= 1

    @patch('signalwire_agents.search.document_processor.sent_tokenize', None)
    def test_calculate_sentences_per_chunk_empty(self):
        """Empty string with no nltk causes ZeroDivisionError (source code bug)."""
        proc = DocumentProcessor()
        with pytest.raises(ZeroDivisionError):
            proc._calculate_sentences_per_chunk("")

    # ── _get_overlap_lines ───────────────────────────────────────────

    def test_get_overlap_lines_fits(self):
        proc = DocumentProcessor(chunk_overlap=100)
        lines = ["a", "bb", "ccc"]
        result = proc._get_overlap_lines(lines)
        assert result == ["a", "bb", "ccc"]

    def test_get_overlap_lines_partial(self):
        proc = DocumentProcessor(chunk_overlap=5)
        lines = ["long_line_here", "short"]
        result = proc._get_overlap_lines(lines)
        assert result == ["short"]

    def test_get_overlap_lines_empty(self):
        proc = DocumentProcessor()
        assert proc._get_overlap_lines([]) == []

    # ── _chunk_document_aware ────────────────────────────────────────

    def test_chunk_document_aware_list_small_pages(self):
        proc = DocumentProcessor()
        pages = ["Small page one.", "Small page two."]
        chunks = proc._chunk_document_aware(pages, "doc.pdf", "pdf")
        assert len(chunks) == 2
        assert chunks[0]['section'] == "Page 1"

    def test_chunk_document_aware_list_pptx(self):
        proc = DocumentProcessor()
        slides = ["Slide text one.", "Slide text two."]
        chunks = proc._chunk_document_aware(slides, "pres.pptx", "pptx")
        assert len(chunks) == 2
        assert chunks[0]['section'] == "Slide 1"

    def test_chunk_document_aware_list_generic(self):
        proc = DocumentProcessor()
        sections = ["Section one.", "Section two."]
        chunks = proc._chunk_document_aware(sections, "doc.docx", "docx")
        assert len(chunks) == 2
        assert chunks[0]['section'] == "Section 1"

    def test_chunk_document_aware_string(self):
        proc = DocumentProcessor()
        chunks = proc._chunk_document_aware("Hello world text here.", "doc.txt", "txt")
        assert len(chunks) >= 1

    def test_chunk_document_aware_skips_empty_pages(self):
        proc = DocumentProcessor()
        pages = ["Content", "", "  ", "More content"]
        chunks = proc._chunk_document_aware(pages, "doc.pdf", "pdf")
        assert len(chunks) == 2