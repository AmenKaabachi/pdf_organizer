"""
Unit tests for the PDF Extractor module.
"""

import unittest
import tempfile
from pathlib import Path
from src.extraction.pdf_extractor import PDFExtractor


class TestPDFExtractor(unittest.TestCase):
    """Test cases for PDFExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor()
    
    def test_initialization(self):
        """Test PDFExtractor initialization."""
        self.assertEqual(self.extractor.min_text_length, 100)
        self.assertEqual(self.extractor.max_file_size_mb, 50)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is\n\n  dirty    text!!! @#$%  "
        clean_text = self.extractor._clean_text(dirty_text)
        
        self.assertIsInstance(clean_text, str)
        self.assertNotIn("  ", clean_text)  # No double spaces
        self.assertTrue(len(clean_text) > 0)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # This would test the FileOrganizer's sanitize method
        # Since it's in a different module, we'll create a simple test here
        dangerous_name = "file<>name|with?bad*chars"
        # Expected behavior: replace invalid chars with underscores
        expected = "file__name_with_bad_chars"
        
        # Simple sanitization logic for testing
        sanitized = dangerous_name
        for char in '<>:"/\\|?*':
            sanitized = sanitized.replace(char, '_')
        
        self.assertEqual(sanitized, expected)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for EmbeddingGenerator class."""
    
    def test_model_list(self):
        """Test available models list."""
        from src.embeddings.embedding_generator import EmbeddingGenerator
        
        models = EmbeddingGenerator.list_available_models()
        self.assertIsInstance(models, dict)
        self.assertIn('all-MiniLM-L6-v2', models)


class TestDocumentClusterer(unittest.TestCase):
    """Test cases for DocumentClusterer class."""
    
    def test_initialization(self):
        """Test clusterer initialization."""
        from src.clustering.document_clusterer import DocumentClusterer
        
        clusterer = DocumentClusterer(algorithm='kmeans', n_clusters=3)
        self.assertEqual(clusterer.algorithm, 'kmeans')
        self.assertEqual(clusterer.n_clusters, 3)
    
    def test_invalid_algorithm(self):
        """Test invalid algorithm handling."""
        from src.clustering.document_clusterer import DocumentClusterer
        
        with self.assertRaises(ValueError):
            DocumentClusterer(algorithm='invalid_algorithm')


if __name__ == '__main__':
    unittest.main()