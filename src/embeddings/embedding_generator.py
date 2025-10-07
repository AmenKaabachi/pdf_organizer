"""
Document embedding generation using Sentence-BERT models.

This module provides functionality to generate semantic embeddings from text content,
supporting various pre-trained models and batch processing capabilities.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates semantic embeddings using Sentence-BERT models."""
    
    # Available pre-trained models
    AVAILABLE_MODELS = {
        # === ENGLISH-ONLY MODELS ===
        'all-MiniLM-L6-v2': {
            'size': 384,
            'description': 'Fast and efficient, good balance of speed and quality (English)',
            'best_for': 'General purpose, large English datasets',
            'languages': 'English'
        },
        'all-mpnet-base-v2': {
            'size': 768,
            'description': 'High quality embeddings, slower but more accurate (English)',
            'best_for': 'High accuracy requirements for English documents',
            'languages': 'English'
        },
        'all-distilroberta-v1': {
            'size': 768,
            'description': 'Good balance of speed and accuracy (English)',
            'best_for': 'Medium-sized English datasets',
            'languages': 'English'
        },
        
        # === TOP 3 MULTILINGUAL MODELS (BEST QUALITY) ===
        'BAAI/bge-m3': {
            'size': 1024,
            'description': '🏆 State-of-the-art multilingual model (100+ languages) - BEST OVERALL',
            'best_for': 'Highest quality multilingual clustering, cross-lingual search',
            'languages': '100+ languages including Arabic, Chinese, French, German, Spanish, Japanese, Russian, Korean, etc.'
        },
        'intfloat/multilingual-e5-large': {
            'size': 1024,
            'description': '🥇 Exceptional multilingual model (100+ languages) - HIGHEST ACCURACY',
            'best_for': 'Maximum quality for diverse multilingual documents',
            'languages': '100+ languages including Arabic, Chinese, French, German, Spanish, Japanese, Russian, Korean, etc.'
        },
        'paraphrase-multilingual-mpnet-base-v2': {
            'size': 768,
            'description': '🥈 High-quality multilingual model (50+ languages) - BEST BALANCE',
            'best_for': 'Fast processing with excellent multilingual quality',
            'languages': '50+ languages including Arabic, Chinese, French, German, Spanish, Japanese, Russian, etc.'
        },
        
        # === ADDITIONAL MULTILINGUAL OPTIONS ===
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'size': 384,
            'description': 'Fast multilingual model - 50+ languages',
            'best_for': 'Large multilingual datasets where speed is critical',
            'languages': '50+ languages including Arabic, English, French, German, Spanish, Chinese, Japanese, Russian'
        },
        'sentence-transformers/distiluse-base-multilingual-cased-v2': {
            'size': 512,
            'description': 'Balanced multilingual model - 15+ major languages',
            'best_for': 'Multilingual semantic similarity with good speed',
            'languages': '15+ languages including Arabic, English, French, German, Spanish, Italian, Dutch, Polish, Turkish, Chinese'
        }
    }
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: Optional[str] = None,
                 cache_embeddings: bool = True,
                 cache_dir: str = './cache/embeddings'):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_embeddings: Whether to cache generated embeddings
            cache_dir: Directory to store cached embeddings
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence-BERT model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding size: {self.embedding_size}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def generate_embeddings(self, 
                          texts: Union[str, List[str]], 
                          batch_size: int = 32,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Check cache first
        if self.cache_embeddings:
            cached_embeddings = self._load_from_cache(texts)
            if cached_embeddings is not None:
                logger.info("Loaded embeddings from cache")
                return cached_embeddings
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for better similarity computation
            )
            
            # Cache embeddings
            if self.cache_embeddings:
                self._save_to_cache(texts, embeddings)
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_document_embeddings(self, 
                                   documents: List[Dict[str, any]], 
                                   text_field: str = 'full_text',
                                   **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for a list of document dictionaries.
        
        Args:
            documents: List of document dictionaries from PDF extraction
            text_field: Field name containing text content
            **kwargs: Additional arguments for generate_embeddings
            
        Returns:
            Tuple of (embeddings array, list of document filenames)
        """
        # Extract texts and filenames
        texts = []
        filenames = []
        
        for doc in documents:
            if doc.get('success', False) and doc.get(text_field):
                # Use first 2000 words to capture more specialized content
                # This includes intro + body content where domain-specific terms appear
                text = ' '.join(doc[text_field].split()[:2000])
                if len(text.strip()) > 0:
                    texts.append(text)
                    filenames.append(doc.get('filename', 'unknown'))
        
        if not texts:
            logger.warning("No valid texts found for embedding generation")
            return np.array([]), []
        
        logger.info(f"Generating embeddings for {len(texts)} valid documents")
        embeddings = self.generate_embeddings(texts, **kwargs)
        
        return embeddings, filenames
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for embeddings.
        
        Args:
            embeddings: Array of normalized embeddings
            
        Returns:
            Similarity matrix (n_docs x n_docs)
        """
        if embeddings.size == 0:
            return np.array([])
        
        # Compute cosine similarity (dot product for normalized vectors)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Ensure diagonal is 1.0 (account for floating point errors)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def find_most_similar(self, 
                         query_text: str, 
                         document_embeddings: np.ndarray,
                         document_names: List[str],
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar documents to a query text.
        
        Args:
            query_text: Text to find similar documents for
            document_embeddings: Pre-computed document embeddings
            document_names: Names/identifiers for documents
            top_k: Number of top similar documents to return
            
        Returns:
            List of (document_name, similarity_score) tuples
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings(query_text)
        
        # Compute similarities
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append((document_names[idx], float(similarities[idx])))
        
        return results
    
    def _generate_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts."""
        import hashlib
        
        # Create hash from model name and texts
        content = f"{self.model_name}_{str(texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, texts: List[str]) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        try:
            cache_key = self._generate_cache_key(texts)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load from cache: {str(e)}")
        
        return None
    
    def _save_to_cache(self, texts: List[str], embeddings: np.ndarray):
        """Save embeddings to cache."""
        try:
            cache_key = self._generate_cache_key(texts)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
                
            logger.debug(f"Saved embeddings to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save to cache: {str(e)}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        model_info = self.AVAILABLE_MODELS.get(self.model_name, {})
        return {
            'model_name': self.model_name,
            'embedding_size': self.embedding_size,
            'device': self.device,
            **model_info
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, any]]:
        """List all available pre-trained models."""
        return cls.AVAILABLE_MODELS


def generate_embeddings(texts: Union[str, List[str]], 
                       model_name: str = 'all-MiniLM-L6-v2',
                       **kwargs) -> np.ndarray:
    """
    Convenience function to generate embeddings.
    
    Args:
        texts: Text(s) to generate embeddings for
        model_name: Name of the model to use
        **kwargs: Additional arguments for EmbeddingGenerator
        
    Returns:
        Array of embeddings
    """
    generator = EmbeddingGenerator(model_name=model_name, **kwargs)
    return generator.generate_embeddings(texts)