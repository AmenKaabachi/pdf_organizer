"""
Document embedding generation using hybrid API/Local approach.

This module provides functionality to generate semantic embeddings from text content,
using APIs for models with free tiers and local downloads for models without free APIs.
"""

import numpy as np
import requests
import json
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import time
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers for local models
try:
    from sentence_transformers import SentenceTransformer
    import torch
    import pickle
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers not available. Local models will be disabled.")

@dataclass
class APIConfig:
    """Configuration for API providers."""
    name: str
    base_url: str
    api_key_env: str
    max_tokens: int
    embedding_size: int
    cost_per_1k: float  # USD per 1K tokens

class EmbeddingGenerator:
    """Generates semantic embeddings using hybrid API/Local approach."""
    
    # Available models - APIs for free options, local downloads for others
    AVAILABLE_MODELS = {
        # === FREE API MODELS (NO DOWNLOADS NEEDED) ===
        'free-hf-all-MiniLM-L6-v2': {
            'provider': 'huggingface_free',
            'size': 384,
            'description': '🆓 FREE HuggingFace API - No downloads, no API key needed',
            'best_for': 'Testing, small datasets, no setup required',
            'languages': 'English',
            'cost_per_1k_tokens': 0.0,  # FREE!
            'api_key_env': None,
            'uses_local': False
        },
        'free-hf-paraphrase-multilingual-mpnet-base-v2': {
            'provider': 'huggingface_free',
            'size': 768,
            'description': '� FREE Multilingual HuggingFace API - No downloads needed',
            'best_for': 'Multilingual testing, no setup required',
            'languages': '50+ languages',
            'cost_per_1k_tokens': 0.0,  # FREE!
            'api_key_env': None,
            'uses_local': False
        },
        
        # === RECOMMENDED: SMALL LOCAL MODELS (FAST DOWNLOADS) ===
        'all-MiniLM-L6-v2': {
            'provider': 'local',
            'size': 384,
            'description': 'RECOMMENDED: Fast and efficient (80MB download)',
            'best_for': 'Default choice - reliable offline processing',
            'languages': 'English',
            'download_size': '80MB',
            'uses_local': True
        },
        'all-mpnet-base-v2': {
            'provider': 'local',
            'size': 768,
            'description': 'Local: High quality embeddings (420MB download)',
            'best_for': 'High accuracy requirements for English documents',
            'languages': 'English',
            'download_size': '420MB',
            'uses_local': True
        },
        'all-distilroberta-v1': {
            'provider': 'local',
            'size': 768,
            'description': '💾 Local: Good balance of speed and accuracy (290MB download)',
            'best_for': 'Medium-sized English datasets',
            'languages': 'English',
            'download_size': '290MB',
            'uses_local': True
        },
        
        # === TOP MULTILINGUAL MODELS (LOCAL DOWNLOADS) ===
        'BAAI/bge-m3': {
            'provider': 'local',
            'size': 1024,
            'description': '�🏆 Local: State-of-the-art multilingual (2.27GB download) - BEST OVERALL',
            'best_for': 'Highest quality multilingual clustering, cross-lingual search',
            'languages': '100+ languages including Arabic, Chinese, French, German, Spanish, Japanese, Russian, Korean, etc.',
            'download_size': '2.27GB',
            'uses_local': True
        },
        'intfloat/multilingual-e5-large': {
            'provider': 'local',
            'size': 1024,
            'description': '💾🥇 Local: Exceptional multilingual (2.24GB download) - HIGHEST ACCURACY',
            'best_for': 'Maximum quality for diverse multilingual documents',
            'languages': '100+ languages including Arabic, Chinese, French, German, Spanish, Japanese, Russian, Korean, etc.',
            'download_size': '2.24GB',
            'uses_local': True
        },
        'paraphrase-multilingual-mpnet-base-v2': {
            'provider': 'local',
            'size': 768,
            'description': '💾🥈 Local: High-quality multilingual (1.11GB download) - BEST BALANCE',
            'best_for': 'Fast processing with excellent multilingual quality',
            'languages': '50+ languages including Arabic, Chinese, French, German, Spanish, Japanese, Russian, etc.',
            'download_size': '1.11GB',
            'uses_local': True
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'provider': 'local',
            'size': 384,
            'description': '💾 Local: Fast multilingual (420MB download)',
            'best_for': 'Large multilingual datasets where speed is critical',
            'languages': '50+ languages including Arabic, English, French, German, Spanish, Chinese, Japanese, Russian',
            'download_size': '420MB',
            'uses_local': True
        },
        'sentence-transformers/distiluse-base-multilingual-cased-v2': {
            'provider': 'local',
            'size': 512,
            'description': '💾 Local: Balanced multilingual (540MB download)',
            'best_for': 'Multilingual semantic similarity with good speed',
            'languages': '15+ languages including Arabic, English, French, German, Spanish, Italian, Dutch, Polish, Turkish, Chinese',
            'download_size': '540MB',
            'uses_local': True
        },
        
        # === BACKUP: FREE API MODELS (MAY BE UNRELIABLE) ===
        'free-hf-all-MiniLM-L6-v2': {
            'provider': 'huggingface_free',
            'size': 384,
            'description': 'BACKUP: FREE HuggingFace API - May have access restrictions',
            'best_for': 'Backup option if local models fail',
            'languages': 'English',
            'cost_per_1k_tokens': 0.0,  # FREE!
            'api_key_env': None,
            'uses_local': False
        },
        'free-hf-paraphrase-multilingual-mpnet-base-v2': {
            'provider': 'huggingface_free',
            'size': 768,
            'description': 'BACKUP: FREE Multilingual HuggingFace API - May be unreliable',
            'best_for': 'Backup multilingual option',
            'languages': '50+ languages',
            'cost_per_1k_tokens': 0.0,  # FREE!
            'api_key_env': None,
            'uses_local': False
        }
    }
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 device: Optional[str] = None, cache_embeddings: bool = True):
        """
        Initialize the hybrid embedding generator.
        
        Args:
            model_name: Name of the model to use (API or local)
            device: Device for local models ('cuda', 'cpu', or None for auto)
            cache_embeddings: Whether to cache embeddings (recommended)
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path("cache/embeddings") if cache_embeddings else None
        
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate model
        if model_name not in self.AVAILABLE_MODELS:
            available = list(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model '{model_name}' not available. Choose from: {available}")
        
        self.model_config = self.AVAILABLE_MODELS[model_name]
        self.provider = self.model_config['provider']
        self.uses_local = self.model_config.get('uses_local', False)
        
        # Initialize based on provider type
        if self.uses_local:
            # LOCAL MODEL SETUP
            if not LOCAL_MODELS_AVAILABLE:
                raise ImportError(
                    f"Local model '{model_name}' requires sentence-transformers. "
                    f"Install with: pip install sentence-transformers"
                )
            
            # Set device for local models
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            
            logger.info(f"💾 Using LOCAL model: {model_name}")
            logger.info(f"📱 Device: {self.device}")
            logger.info(f"📦 Download size: {self.model_config.get('download_size', 'Unknown')}")
            
            # Load local model
            self._load_local_model()
            
        else:
            # API MODEL SETUP
            self.api_key = None
            api_key_env = self.model_config.get('api_key_env')
            if api_key_env:
                self.api_key = os.getenv(api_key_env)
                if not self.api_key:
                    logger.warning(f"⚠️ API key not found in environment variable '{api_key_env}'. "
                                 f"Set it with: export {api_key_env}=your_api_key")
            
            logger.info(f"🌐 Using API model: {model_name}")
            if 'cost_per_1k_tokens' in self.model_config:
                logger.info(f"💰 Cost: ${self.model_config['cost_per_1k_tokens']:.4f} per 1K tokens")
            else:
                logger.info(f"💰 Cost: FREE!")
    
    def _load_local_model(self):
        """Load the Sentence-BERT model for local inference."""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Local model loaded successfully. Embedding size: {self.embedding_size}")
        except Exception as e:
            logger.error(f"❌ Failed to load local model {self.model_name}: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        if self.cache_embeddings:
            cached_embeddings = self._load_from_cache(cache_key)
            if cached_embeddings is not None:
                logger.info(f"📦 Loaded {len(texts)} embeddings from cache")
                return cached_embeddings
        
        # Generate using appropriate method
        if self.uses_local:
            embeddings = self._generate_local_embeddings(texts, batch_size, show_progress)
        else:
            embeddings = self._generate_api_embeddings(texts, batch_size, show_progress)
        
        # Cache results
        if self.cache_embeddings:
            self._save_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def _generate_local_embeddings(self, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """Generate embeddings using local model."""
        try:
            logger.info(f"💾 Generating {len(texts)} embeddings locally...")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for better similarity computation
            )
            
            logger.info(f"✅ Generated local embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating local embeddings: {str(e)}")
            raise
    
    def _generate_api_embeddings(self, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """Generate embeddings using API."""
        logger.info(f"🌐 Generating {len(texts)} embeddings via {self.provider.upper()} API...")
        
        # Process in batches
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if show_progress:
                progress = (i + len(batch)) / len(texts) * 100
                print(f"API Progress: {progress:.1f}% ({i + len(batch)}/{len(texts)})")
            
            # Get embeddings for this batch
            batch_embeddings, batch_tokens = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)
            total_tokens += batch_tokens
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        embeddings = np.array(all_embeddings)
        
        # Log cost information
        if 'cost_per_1k_tokens' in self.model_config:
            cost = (total_tokens / 1000) * self.model_config['cost_per_1k_tokens']
            logger.info(f"💰 Tokens used: {total_tokens:,} (≈${cost:.4f})")
        else:
            logger.info(f"💰 Tokens used: {total_tokens:,} (FREE!)")
        
        logger.info(f"✅ Generated API embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _call_api(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Call the appropriate API based on provider."""
        if self.provider == 'openai':
            return self._call_openai_api(texts)
        elif self.provider == 'cohere':
            return self._call_cohere_api(texts)
        elif self.provider == 'huggingface':
            return self._call_huggingface_api(texts)
        elif self.provider == 'huggingface_free':
            return self._call_huggingface_free_api(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _call_openai_api(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Call OpenAI Embeddings API."""
        import openai
        
        openai.api_key = self.api_key
        
        # Map model names
        model_map = {
            'openai-text-embedding-3-small': 'text-embedding-3-small',
            'openai-text-embedding-3-large': 'text-embedding-3-large'
        }
        
        model_id = model_map[self.model_name]
        
        try:
            response = openai.embeddings.create(
                input=texts,
                model=model_id
            )
            
            embeddings = [item.embedding for item in response.data]
            tokens = response.usage.total_tokens
            
            return embeddings, tokens
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            raise
    
    def _call_cohere_api(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Call Cohere Embeddings API."""
        url = "https://api.cohere.ai/v1/embed"
        
        # Map model names
        model_map = {
            'cohere-embed-multilingual-v3': 'embed-multilingual-v3.0',
            'cohere-embed-english-v3': 'embed-english-v3.0'
        }
        
        model_id = model_map[self.model_name]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "texts": texts,
            "model": model_id,
            "input_type": "search_document"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            embeddings = result["embeddings"]
            
            # Approximate token count
            tokens = sum(len(text.split()) for text in texts)
            
            return embeddings, tokens
            
        except Exception as e:
            logger.error(f"❌ Cohere API error: {e}")
            raise
    
    def _call_huggingface_api(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Call HuggingFace Inference API (paid)."""
        # Extract model ID from our model name
        model_id = self.model_name.replace('hf-', '')
        
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            embeddings = response.json()
            
            # Approximate token count
            tokens = sum(len(text.split()) for text in texts)
            
            return embeddings, tokens
            
        except Exception as e:
            logger.error(f"❌ HuggingFace API error: {e}")
            raise
    
    def _call_huggingface_free_api(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Call HuggingFace Inference API (free tier)."""
        # Map our model names to HuggingFace model IDs
        model_map = {
            'free-hf-all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
            'free-hf-paraphrase-multilingual-mpnet-base-v2': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        }
        
        model_id = model_map.get(self.model_name, 'sentence-transformers/all-MiniLM-L6-v2')
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {"Content-Type": "application/json"}
        data = {"inputs": texts}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 401:
                # Try with minimal rate limiting for public access
                logger.warning("⚠️ HuggingFace API returned 401. This may be due to rate limits or policy changes.")
                logger.info("💡 Consider using local models instead. They work offline and have no API restrictions.")
                raise Exception(f"HuggingFace Free API access denied. Use local models instead.")
            
            if response.status_code == 503:
                logger.info("⏳ Model loading, waiting 20 seconds...")
                time.sleep(20)
                response = requests.post(url, headers=headers, json=data)
            
            response.raise_for_status()
            embeddings = response.json()
            
            # Validate response format
            if not isinstance(embeddings, list) or not embeddings:
                raise ValueError("Invalid response format from HuggingFace API")
            
            # Approximate token count
            tokens = sum(len(text.split()) for text in texts)
            
            return embeddings, tokens
            
        except Exception as e:
            logger.error(f"❌ HuggingFace Free API error: {e}")
            logger.info("💡 Suggestion: Use local models for reliable offline access")
            raise
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts."""
        import hashlib
        content = f"{self.model_name}:{':'.join(texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache."""
        if self.uses_local:
            # Use pickle format for local models (backward compatibility)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load local cache: {e}")
        else:
            # Use numpy format for API models
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    return np.load(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to load API cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """Save embeddings to cache."""
        try:
            if self.uses_local:
                # Use pickle format for local models (backward compatibility)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                logger.debug(f"Saved local embeddings to cache: {cache_file}")
            else:
                # Use numpy format for API models
                cache_file = self.cache_dir / f"{cache_key}.npy"
                np.save(cache_file, embeddings)
                logger.debug(f"Saved API embeddings to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def generate_document_embeddings(self, extraction_results: List[Dict], 
                                   batch_size: int = 16, show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for extracted document results.
        
        Args:
            extraction_results: List of PDF extraction results (or documents list)
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            Tuple of (embeddings array, document names list)
        """
        # Extract texts and names
        texts = []
        document_names = []
        
        for result in extraction_results:
            # Handle both old format (extraction_results) and new format (documents)
            if isinstance(result, dict):
                if result.get('success', False) and result.get('full_text'):
                    # Old extraction format
                    text = result['full_text']
                    filename = result['filename']
                elif result.get('full_text'):
                    # New document format
                    text = result['full_text']
                    filename = result.get('filename', 'unknown')
                else:
                    continue
                
                # Limit text length - more for local models, less for APIs
                words = text.split()
                if self.uses_local:
                    # Local models can handle more text (first 2000 words)
                    max_words = 2000
                else:
                    # API models should use less to avoid costs (first 1500 words)
                    max_words = 1500
                
                if len(words) > max_words:
                    text = ' '.join(words[:max_words])
                
                if len(text.strip()) > 0:
                    texts.append(text)
                    document_names.append(filename)
        
        if not texts:
            logger.warning("No valid texts found for embedding generation")
            return np.array([]), []
        
        logger.info(f"Generating embeddings for {len(texts)} valid documents")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size, show_progress)
        
        return embeddings, document_names
    
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
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings)
        except ImportError:
            # Fallback: manual cosine similarity computation
            # Normalize embeddings if not already normalized
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_embeddings = embeddings / norms
            
            # Compute cosine similarity (dot product for normalized vectors)
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
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
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        model_info = self.model_config.copy()
        
        # Add runtime info
        if self.uses_local and hasattr(self, 'model'):
            model_info.update({
                'model_name': self.model_name,
                'embedding_size': getattr(self, 'embedding_size', self.model_config.get('size', 'Unknown')),
                'device': getattr(self, 'device', 'Unknown')
            })
        
        return model_info
    
    @classmethod
    def list_available_models(cls) -> Dict:
        """List all available models."""
        return cls.AVAILABLE_MODELS.copy()


# Convenience function for backward compatibility
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