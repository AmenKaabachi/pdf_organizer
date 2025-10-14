# Hybrid Embedding Approach

## Overview

The PDF Organizer now uses a **hybrid approach** for generating embeddings:

- **🆓 Free APIs**: Only for models with truly free and reliable access (currently limited due to HuggingFace policy changes)
- **💾 Local Models**: For all other models - reliable, offline, no API restrictions

## Why This Approach?

1. **Reliability**: Local models work offline without API dependencies
2. **No API Limits**: No rate limiting or access restrictions
3. **Cost Control**: Only models with genuinely free APIs use cloud services
4. **Quality**: Access to state-of-the-art models like `BAAI/bge-m3`

## Model Categories

### 🥇 RECOMMENDED: Small Local Models

- **`all-MiniLM-L6-v2`** (80MB) - Default choice, fast and reliable
- **`all-mpnet-base-v2`** (420MB) - Higher quality English embeddings
- **`all-distilroberta-v1`** (290MB) - Balanced speed and accuracy

### 🌍 Multilingual Local Models

- **`paraphrase-multilingual-MiniLM-L12-v2`** (420MB) - Fast multilingual
- **`paraphrase-multilingual-mpnet-base-v2`** (1.11GB) - Best balance
- **`BAAI/bge-m3`** (2.27GB) - State-of-the-art, 100+ languages
- **`intfloat/multilingual-e5-large`** (2.24GB) - Highest accuracy

### ⚠️ Backup: Free APIs (May Be Unreliable)

- **`free-hf-all-MiniLM-L6-v2`** - Currently has access restrictions
- **`free-hf-paraphrase-multilingual-mpnet-base-v2`** - Currently has access restrictions

## Usage

```python
from src.embeddings.embedding_generator import EmbeddingGenerator

# Default: Reliable local model (80MB download)
generator = EmbeddingGenerator()  # Uses 'all-MiniLM-L6-v2'

# Multilingual: Local model (420MB download)
generator = EmbeddingGenerator('paraphrase-multilingual-MiniLM-L12-v2')

# Best quality: Local model (2.27GB download)
generator = EmbeddingGenerator('BAAI/bge-m3')

# Generate embeddings
embeddings = generator.generate_embeddings(["Text 1", "Text 2"])
```

## Setup Requirements

### For Local Models (Recommended)

```bash
pip install sentence-transformers torch
```

### Minimal Setup (APIs only - currently unreliable)

```bash
# No additional dependencies needed
# But APIs may have access restrictions
```

## First Run

- **Local models**: Downloaded once on first use, then cached
- **API models**: No downloads, but may have access issues
- **Embeddings**: Always cached for faster subsequent runs

## Migration from Pure API Approach

The system automatically detects available dependencies:

- If `sentence-transformers` is installed → Local models available
- If not installed → Only API models (currently limited)

**Recommendation**: Install `sentence-transformers` for reliable operation.
