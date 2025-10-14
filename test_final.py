#!/usr/bin/env python3
"""
Final integration test for the hybrid embedding system.
Tests the default recommended model and basic functionality.
"""

import sys
import os
sys.path.append('src')

def test_default_setup():
    """Test the default setup with recommended model."""
    print("🧪 Testing DEFAULT HYBRID SETUP")
    print("=" * 50)
    
    # Test imports
    try:
        from embeddings.embedding_generator import EmbeddingGenerator
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Check available models
    try:
        models = EmbeddingGenerator.list_available_models()
        print(f"✅ Found {len(models)} available models")
        
        # Count local vs API models
        local_models = sum(1 for m in models.values() if m.get('uses_local', False))
        api_models = sum(1 for m in models.values() if not m.get('uses_local', False))
        print(f"   📱 Local models: {local_models}")
        print(f"   🌐 API models: {api_models}")
        
    except Exception as e:
        print(f"❌ Model listing failed: {e}")
        return False
    
    # Test default model
    try:
        print(f"\n🔬 Testing default model...")
        generator = EmbeddingGenerator()  # Should use 'all-MiniLM-L6-v2'
        
        info = generator.get_model_info()
        print(f"   📊 Model: {generator.model_name}")
        print(f"   💾 Uses local: {info.get('uses_local', 'Unknown')}")
        print(f"   📦 Size: {info.get('download_size', info.get('size', 'Unknown'))}")
        
    except Exception as e:
        print(f"❌ Default model setup failed: {e}")
        return False
    
    # Test embedding generation
    try:
        print(f"\n🧮 Testing embedding generation...")
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
        
        embeddings = generator.generate_embeddings(test_texts, show_progress=False)
        print(f"   ✅ Generated embeddings: {embeddings.shape}")
        
        # Test similarity computation
        similarity_matrix = generator.compute_similarity_matrix(embeddings)
        print(f"   ✅ Similarity matrix: {similarity_matrix.shape}")
        print(f"   📊 Max similarity: {similarity_matrix.max():.3f}")
        print(f"   📊 Min similarity: {similarity_matrix.min():.3f}")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False
    
    # Test document processing format
    try:
        print(f"\n📄 Testing document processing format...")
        mock_documents = [
            {
                'success': True,
                'filename': 'doc1.pdf',
                'full_text': 'This is the first document about machine learning and AI.'
            },
            {
                'success': True, 
                'filename': 'doc2.pdf',
                'full_text': 'This is the second document about natural language processing.'
            }
        ]
        
        doc_embeddings, doc_names = generator.generate_document_embeddings(
            mock_documents, show_progress=False
        )
        
        print(f"   ✅ Document embeddings: {doc_embeddings.shape}")
        print(f"   ✅ Document names: {doc_names}")
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return False
    
    print(f"\n" + "=" * 50)
    print("🎉 DEFAULT SETUP TEST PASSED!")
    print(f"\n💡 SYSTEM STATUS:")
    print(f"   • Default model: {generator.model_name}")
    print(f"   • Processing: {'Local' if info.get('uses_local') else 'API'}")
    print(f"   • Ready for use: ✅")
    
    return True

def test_requirements():
    """Test if all required dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    required_imports = [
        ('numpy', 'Core numerical operations'),
        ('requests', 'API communication'),
        ('pathlib', 'File system operations')
    ]
    
    optional_imports = [
        ('sentence_transformers', 'Local model support'),
        ('torch', 'Neural network backend'),
        ('sklearn', 'Similarity computation'),
        ('streamlit', 'Web interface'),
        ('plotly', 'Visualization')
    ]
    
    print("📦 Required dependencies:")
    for module, description in required_imports:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
        except ImportError:
            print(f"   ❌ {module} - {description} (MISSING)")
    
    print("\n📦 Optional dependencies:")
    for module, description in optional_imports:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
        except ImportError:
            print(f"   ⚠️ {module} - {description} (not installed)")

if __name__ == "__main__":
    print("🚀 PDF ORGANIZER AI - HYBRID SYSTEM TEST")
    print("=" * 60)
    
    # Check dependencies first
    test_requirements()
    
    # Test main functionality
    success = test_default_setup()
    
    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Run Streamlit app: streamlit run src/app/streamlit_app.py")
        print(f"   2. Upload PDF files for clustering")
        print(f"   3. Choose recommended model: all-MiniLM-L6-v2")
        print(f"   4. Enjoy intelligent document organization!")
    else:
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   1. Install sentence-transformers: pip install sentence-transformers")
        print(f"   2. Check Python version: Python 3.8+ required")
        print(f"   3. Review error messages above")
        print(f"   4. See docs/HYBRID_APPROACH.md for details")