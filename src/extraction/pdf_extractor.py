"""
PDF text extraction module using PyMuPDF.

This module provides functionality to extract clean text content from PDF files,
handle various PDF formats, and preprocess text for embedding generation.
"""

import fitz  # PyMuPDF
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Handles PDF text extraction and preprocessing."""
    
    def __init__(self, min_text_length: int = 100, max_file_size_mb: int = 50):
        """
        Initialize PDF extractor.
        
        Args:
            min_text_length: Minimum text length to consider a page valid
            max_file_size_mb: Maximum file size in MB to process
        """
        self.min_text_length = min_text_length
        self.max_file_size_mb = max_file_size_mb
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        # Validate file
        if not self._validate_pdf_file(pdf_path):
            return self._create_empty_result(pdf_path, "Invalid file")
        
        try:
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            # Get page count and metadata before closing document
            total_pages = doc.page_count
            metadata = self._extract_metadata(pdf_path, doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text()
                
                if len(page_text.strip()) >= self.min_text_length:
                    page_texts.append({
                        'page_number': page_num + 1,
                        'text': self._clean_text(page_text)
                    })
                    full_text += page_text + "\n"
            
            # Close document after extraction
            doc.close()
            
            # Clean and process full text
            cleaned_text = self._clean_text(full_text)
            
            return {
                'file_path': str(pdf_path),
                'filename': pdf_path.name,
                'full_text': cleaned_text,
                'page_texts': page_texts,
                'page_count': len(page_texts),
                'total_pages': total_pages,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text),
                'metadata': metadata,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return self._create_empty_result(pdf_path, str(e))
    
    def extract_from_directory(self, directory_path: str, 
                             recursive: bool = True) -> List[Dict[str, any]]:
        """
        Extract text from all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            recursive: Whether to search subdirectories
            
        Returns:
            List of extraction results for each PDF
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find all PDF files
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory_path.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        results = []
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            result = self.extract_text_from_pdf(pdf_file)
            results.append(result)
        
        # Log summary
        successful = len([r for r in results if r['success']])
        logger.info(f"Successfully processed {successful}/{len(pdf_files)} files")
        
        return results
    
    def _validate_pdf_file(self, pdf_path: Path) -> bool:
        """Validate PDF file before processing."""
        try:
            # Check if file exists
            if not pdf_path.exists():
                logger.warning(f"File not found: {pdf_path}")
                return False
            
            # Check file extension
            if pdf_path.suffix.lower() != '.pdf':
                logger.warning(f"Not a PDF file: {pdf_path}")
                return False
            
            # Check file size
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.warning(f"File too large ({file_size_mb:.1f}MB): {pdf_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {pdf_path}: {str(e)}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove very short lines (likely headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        # Join and normalize
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def _extract_metadata(self, pdf_path: Path, doc) -> Dict[str, any]:
        """Extract metadata from PDF document."""
        try:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2),
                'file_extension': pdf_path.suffix.lower()
            }
        except Exception as e:
            logger.warning(f"Could not extract metadata from {pdf_path}: {str(e)}")
            return {}
    
    def _create_empty_result(self, pdf_path: Path, error_msg: str) -> Dict[str, any]:
        """Create empty result for failed extraction."""
        return {
            'file_path': str(pdf_path),
            'filename': pdf_path.name if pdf_path else 'unknown',
            'full_text': '',
            'page_texts': [],
            'page_count': 0,
            'total_pages': 0,
            'word_count': 0,
            'char_count': 0,
            'metadata': {},
            'success': False,
            'error': error_msg
        }


def extract_text_from_pdf(pdf_path: str, **kwargs) -> Dict[str, any]:
    """
    Convenience function to extract text from a single PDF.
    
    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments for PDFExtractor
        
    Returns:
        Extraction result dictionary
    """
    extractor = PDFExtractor(**kwargs)
    return extractor.extract_text_from_pdf(pdf_path)


def extract_from_directory(directory_path: str, **kwargs) -> List[Dict[str, any]]:
    """
    Convenience function to extract text from all PDFs in directory.
    
    Args:
        directory_path: Path to directory
        **kwargs: Additional arguments for PDFExtractor
        
    Returns:
        List of extraction results
    """
    extractor = PDFExtractor(**kwargs)
    return extractor.extract_from_directory(directory_path, **kwargs)