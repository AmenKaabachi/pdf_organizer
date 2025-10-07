"""
Setup script to pre-download required models.
Run this once after installing requirements: python setup_models.py
"""

import os
from sentence_transformers import SentenceTransformer

def download_models():
    """Download and cache all required models."""
    
    print("=" * 60)
    print("PDF Organizer AI - Model Setup")
    print("=" * 60)
    print("\nThis will download the required AI model (~438MB).")
    print("This only needs to be done ONCE.\n")
    
    # Define models to download
    models = [
        {
            'name': 'all-mpnet-base-v2',
            'size': '438MB',
            'description': 'High-quality sentence embedding model'
        }
    ]
    
    for i, model_info in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Downloading {model_info['name']}...")
        print(f"    Size: {model_info['size']}")
        print(f"    Purpose: {model_info['description']}")
        print()
        
        try:
            # Download and cache the model
            model = SentenceTransformer(model_info['name'])
            
            # Get cache location
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
            
            print(f"    ✓ Successfully downloaded!")
            print(f"    ✓ Cached at: {cache_dir}")
            
        except Exception as e:
            print(f"    ✗ Error downloading model: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ All models downloaded successfully!")
    print("=" * 60)
    print("\nYou can now run the application:")
    print("  python -m streamlit run app.py")
    print()
    
    return True

if __name__ == "__main__":
    download_models()
