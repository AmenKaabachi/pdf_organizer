"""
Streamlit web application for the AI-Powered PDF Organizer.
Simple, clean original design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import zipfile
import io
import json
from pathlib import Path
import time
from typing import List, Dict

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from extraction.pdf_extractor import PDFExtractor
from embeddings.embedding_generator import EmbeddingGenerator
from clustering.document_clusterer import DocumentClusterer
from organization.file_organizer import FileOrganizer

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered PDF Organizer",
    page_icon="📄",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    h1 {
        color: #1f77b4;
    }
    
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background-color: #145a8f;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = None
    if 'clusterer' not in st.session_state:
        st.session_state.clusterer = None
    if 'document_names' not in st.session_state:
        st.session_state.document_names = None


def main():
    """Main application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 AI-Powered PDF Organizer")
    st.markdown("**Intelligent Document Management with Machine Learning**")
    st.markdown("---")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Step 1", "Upload PDFs", delta="Extract Text")
    with col2:
        st.metric("Step 2", "AI Analysis", delta="Generate Embeddings")
    with col3:
        st.metric("Step 3", "Clustering", delta="Group Similar Docs")
    with col4:
        st.metric("Step 4", "Organize", delta="Create Folders")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Embedding model selection
        st.subheader("Embedding Model")
        available_models = EmbeddingGenerator.list_available_models()
        
        model_options = list(available_models.keys())
        model_descriptions = [f"{name} - {info['description'][:50]}..." 
                            for name, info in available_models.items()]
        
        selected_model_index = st.selectbox(
            "Choose AI Model",
            range(len(model_options)),
            format_func=lambda i: model_options[i],
            help="Select the Sentence-BERT model for generating embeddings"
        )
        selected_model = model_options[selected_model_index]
        
        # Show model info
        with st.expander("ℹ️ Model Information"):
            model_info = available_models[selected_model]
            st.write(f"**Description:** {model_info['description']}")
            st.write(f"**Embedding Size:** {model_info['embedding_size']}")
            st.write(f"**Performance:** {model_info['performance']}")
        
        st.markdown("---")
        
        # Clustering settings
        st.subheader("Clustering Settings")
        
        clustering_algorithm = st.selectbox(
            "Algorithm",
            ["kmeans", "dbscan", "agglomerative"],
            help="Choose clustering algorithm"
        )
        
        if clustering_algorithm == "kmeans":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            auto_tune = st.checkbox("Auto-tune clusters", value=True)
        elif clustering_algorithm == "dbscan":
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 1, 10, 2)
            n_clusters = None
            auto_tune = False
        else:  # agglomerative
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            auto_tune = False
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Analysis", "🗂️ Organize", "💡 Help"])
    
    with tab1:
        st.header("Upload PDF Files")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to organize"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # Show file list
            with st.expander("📋 Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Start Processing", type="primary"):
                process_documents(uploaded_files, selected_model, clustering_algorithm, 
                                n_clusters, auto_tune, eps if clustering_algorithm == "dbscan" else None,
                                min_samples if clustering_algorithm == "dbscan" else None)
    
    with tab2:
        st.header("Analysis Results")
        
        if st.session_state.extraction_results is None:
            st.info("👆 Upload and process PDF files first")
        else:
            show_analysis_results()
    
    with tab3:
        st.header("Organize Files")
        
        if st.session_state.cluster_labels is None:
            st.info("👆 Process documents first to see organization options")
        else:
            show_organization_options()
    
    with tab4:
        st.header("Help & Information")
        show_help_info()


def process_documents(uploaded_files, model_name, algorithm, n_clusters, auto_tune, eps, min_samples):
    """Process uploaded PDF documents."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Extract text
        status_text.text("📄 Extracting text from PDFs...")
        progress_bar.progress(0.25)
        
        extractor = PDFExtractor()
        extraction_results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_path = Path(temp_dir) / file.name
                temp_path.write_bytes(file.read())
                result = extractor.extract_text_from_pdf(str(temp_path))
                extraction_results.append(result)
        
        st.session_state.extraction_results = extraction_results
        
        # Step 2: Generate embeddings
        status_text.text("🧠 Generating AI embeddings...")
        progress_bar.progress(0.5)
        
        embedding_generator = EmbeddingGenerator(model_name=model_name)
        embeddings, document_names = embedding_generator.generate_document_embeddings(
            extraction_results,
            batch_size=16
        )
        
        st.session_state.embeddings = embeddings
        st.session_state.document_names = document_names
        
        # Step 3: Cluster documents
        status_text.text("🎯 Clustering documents...")
        progress_bar.progress(0.75)
        
        clusterer_kwargs = {'algorithm': algorithm, 'random_state': 42}
        
        if algorithm == "kmeans":
            clusterer_kwargs['n_clusters'] = n_clusters
        elif algorithm == "dbscan":
            clusterer_kwargs['eps'] = eps
            clusterer_kwargs['min_samples'] = min_samples
        else:  # agglomerative
            clusterer_kwargs['n_clusters'] = n_clusters
        
        clusterer = DocumentClusterer(**clusterer_kwargs)
        cluster_labels = clusterer.fit_predict(embeddings, auto_tune=auto_tune)
        
        st.session_state.cluster_labels = cluster_labels
        st.session_state.clusterer = clusterer
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"🎉 Successfully processed {len(uploaded_files)} documents into {len(set(cluster_labels))} clusters!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def show_analysis_results():
    """Display analysis results."""
    
    extraction_results = st.session_state.extraction_results
    embeddings = st.session_state.embeddings
    cluster_labels = st.session_state.cluster_labels
    clusterer = st.session_state.clusterer
    document_names = st.session_state.document_names
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", len(extraction_results))
    
    with col2:
        successful = sum(1 for r in extraction_results if r['success'])
        st.metric("✅ Successfully Processed", successful)
    
    with col3:
        st.metric("🎯 Clusters Found", len(set(cluster_labels)))
    
    with col4:
        if clusterer and hasattr(clusterer, 'silhouette_score'):
            st.metric("📊 Quality Score", f"{clusterer.silhouette_score:.3f}")
    
    st.markdown("---")
    
    # Document details
    st.subheader("📋 Document Details")
    
    df_data = []
    for i, (result, cluster) in enumerate(zip(extraction_results, cluster_labels)):
        df_data.append({
            'Document': result['filename'],
            'Cluster': f"Cluster {cluster}",
            'Pages': result['page_count'],
            'Words': result['word_count'],
            'Status': '✅' if result['success'] else '❌'
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Visualization
    st.subheader("📊 Cluster Visualization")
    
    # Use PCA for 2D visualization
    from sklearn.decomposition import PCA
    
    if len(embeddings) >= 2:
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            color=[f"Cluster {c}" for c in cluster_labels],
            text=document_names,
            title="Document Clusters (PCA Projection)",
            labels={'x': 'PC1', 'y': 'PC2'}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 documents for visualization")


def show_organization_options():
    """Show file organization options."""
    
    cluster_labels = st.session_state.cluster_labels
    document_names = st.session_state.document_names
    
    st.subheader("🗂️ Organization Preview")
    
    # Show clusters
    for cluster_id in sorted(set(cluster_labels)):
        cluster_docs = [doc for doc, label in zip(document_names, cluster_labels) if label == cluster_id]
        
        with st.expander(f"📁 Cluster {cluster_id} ({len(cluster_docs)} documents)"):
            for doc in cluster_docs:
                st.write(f"- {doc}")
    
    st.markdown("---")
    
    # Organization options
    st.subheader("📦 Download Organized Files")
    
    output_format = st.radio(
        "Organization Format",
        ["Separate ZIP files per cluster", "Single ZIP with folders"],
        help="Choose how to package the organized files"
    )
    
    if st.button("📥 Download Organized Files"):
        st.info("💡 This would create organized output. Feature requires actual file handling.")


def show_help_info():
    """Display help and information."""
    
    st.markdown("""
    ## 🤖 How It Works
    
    This application uses state-of-the-art AI to automatically organize your PDF documents:
    
    ### 📄 Step 1: Text Extraction
    - Extracts clean text from PDF files
    - Handles multi-page documents
    - Preserves document structure
    
    ### 🧠 Step 2: AI Embeddings
    - Uses Sentence-BERT models
    - Converts documents to semantic vectors
    - Captures meaning and context
    
    ### 🎯 Step 3: Clustering
    - Groups similar documents together
    - Multiple algorithms available (K-Means, DBSCAN, Agglomerative)
    - Automatic parameter tuning
    
    ### 🗂️ Step 4: Organization
    - Creates themed folders
    - Copies/moves files automatically
    - Generates detailed reports
    
    ## 🔧 Technologies Used
    
    - **PyMuPDF**: PDF text extraction
    - **Sentence-Transformers**: AI embeddings
    - **Scikit-learn**: Machine learning clustering
    - **Streamlit**: Web interface
    
    ## 💡 Tips
    
    - Upload PDFs with substantial text content
    - Use 3-5 clusters for small collections
    - Try auto-tune for optimal results
    - Different models work better for different document types
    
    ## 📚 Model Information
    
    **all-MiniLM-L6-v2**: Fast, efficient, good for general purposes  
    **all-mpnet-base-v2**: Higher quality, best overall performance  
    **all-distilroberta-v1**: Balanced speed and quality  
    **paraphrase-MiniLM-L6-v2**: Optimized for semantic similarity  
    """)


if __name__ == "__main__":
    main()
