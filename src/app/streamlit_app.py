# -*- coding: utf-8 -*-
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
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'processing_cancelled' not in st.session_state:
        st.session_state.processing_cancelled = False


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
        st.subheader("🌍 Embedding Model")
        available_models = EmbeddingGenerator.list_available_models()
        
        model_options = list(available_models.keys())
        
        # Group models by language support
        english_models = [name for name, info in available_models.items() if info.get('languages', '').startswith('English')]
        multilingual_models = [name for name, info in available_models.items() if 'languages' in info and not info['languages'].startswith('English')]
        
        # Default to BAAI/bge-m3 (state-of-the-art multilingual model)
        default_model = 'BAAI/bge-m3'
        default_index = model_options.index(default_model) if default_model in model_options else 0
        
        selected_model_index = st.selectbox(
            "Choose AI Model",
            range(len(model_options)),
            index=default_index,  # Default to multilingual model
            format_func=lambda i: f"{'🌍 [Multi]' if model_options[i] in multilingual_models else '[EN]'} {model_options[i]}",
            help="Select the Sentence-BERT model. 🌍 = Multilingual (50+ languages), [EN] = English only"
        )
        selected_model = model_options[selected_model_index]
        
        # Show model info with language support highlighted
        with st.expander("ℹ️ Model Information"):
            model_info = available_models[selected_model]
            st.write(f"**Description:** {model_info['description']}")
            st.write(f"**Embedding Size:** {model_info['size']}")
            st.write(f"**Best For:** {model_info['best_for']}")
            
            # Highlight language support
            if 'languages' in model_info:
                if model_info['languages'].startswith('English'):
                    st.info(f"[EN] **Languages:** {model_info['languages']}")
                else:
                    st.success(f"🌍 **Languages:** {model_info['languages']}")
        
        # Show language support summary
        if selected_model in multilingual_models:
            st.success("✅ Multilingual model selected - works with documents in any language!")
        else:
            st.info("ℹ️ English-only model selected")
        
        st.markdown("---")
        
        # Clustering settings
        st.subheader("Clustering Settings")
        
        clustering_algorithm = st.selectbox(
            "Algorithm",
            ["kmeans", "dbscan", "agglomerative"],
            help="Choose clustering algorithm"
        )
        
        if clustering_algorithm == "kmeans":
            # Auto-tune checkbox FIRST
            auto_tune = st.checkbox("Auto-tune clusters", value=True, 
                                   help="Automatically find optimal number of clusters")
            
            # Manual slider (disabled when auto-tune is on)
            n_clusters = st.slider(
                "Number of Clusters", 
                2, 10, 3,
                disabled=auto_tune,  # Grey out when auto-tune is enabled
                help="Manual cluster count (disabled when auto-tune is on)"
            )
            
            if auto_tune:
                st.info("🤖 Auto-tune enabled: Optimal cluster count will be determined automatically")
        
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
            
            # Show Start or Cancel button based on processing state
            col_btn1, col_btn2 = st.columns([1, 4])
            
            with col_btn1:
                if not st.session_state.processing:
                    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                        st.session_state.processing = True
                        st.session_state.processing_cancelled = False
                        st.rerun()
                else:
                    if st.button("🛑 Cancel", type="secondary", use_container_width=True):
                        st.session_state.processing_cancelled = True
                        st.warning("⚠️ Cancelling process...")
                        time.sleep(0.5)
                        st.session_state.processing = False
                        st.rerun()
            
            with col_btn2:
                if st.session_state.processing:
                    st.info("⏳ Processing in progress... Click Cancel to stop and change settings.")
            
            # Process if flag is set
            if st.session_state.processing and not st.session_state.processing_cancelled:
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
        # Check for cancellation
        if st.session_state.processing_cancelled:
            status_text.text("🛑 Processing cancelled")
            progress_bar.empty()
            st.session_state.processing = False
            return
        
        # Save uploaded file data for later download
        uploaded_file_data = {}
        for file in uploaded_files:
            file.seek(0)  # Reset file pointer
            uploaded_file_data[file.name] = file.read()
        
        st.session_state.uploaded_file_data = uploaded_file_data
        
        # Step 1: Extract text
        status_text.text("📄 Extracting text from PDFs...")
        progress_bar.progress(0.25)
        
        # Check for cancellation
        if st.session_state.processing_cancelled:
            status_text.text("🛑 Processing cancelled")
            progress_bar.empty()
            st.session_state.processing = False
            return
        
        extractor = PDFExtractor()
        extraction_results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_name, file_data in uploaded_file_data.items():
                temp_path = Path(temp_dir) / file_name
                temp_path.write_bytes(file_data)
                result = extractor.extract_text_from_pdf(str(temp_path))
                extraction_results.append(result)
        
        st.session_state.extraction_results = extraction_results
        
        # Check for cancellation
        if st.session_state.processing_cancelled:
            status_text.text("🛑 Processing cancelled after extraction")
            progress_bar.empty()
            st.session_state.processing = False
            return
        
        # Step 2: Generate embeddings
        status_text.text(f"🧠 Generating AI embeddings with {model_name}...")
        progress_bar.progress(0.5)
        
        # Use the model selected by the user from the dropdown
        embedding_generator = EmbeddingGenerator(model_name=model_name)
        embeddings, document_names = embedding_generator.generate_document_embeddings(
            extraction_results,
            batch_size=16
        )
        
        st.session_state.embeddings = embeddings
        st.session_state.document_names = document_names
        
        # Check for cancellation
        if st.session_state.processing_cancelled:
            status_text.text("🛑 Processing cancelled after embeddings")
            progress_bar.empty()
            st.session_state.processing = False
            return
        
        # Step 3: Cluster documents
        status_text.text("🎯 Clustering documents...")
        progress_bar.progress(0.75)
        
        clusterer_kwargs = {'algorithm': algorithm, 'random_state': 42}
        
        # IMPORTANT: Only pass n_clusters if NOT using auto-tune
        if algorithm == "kmeans":
            if not auto_tune:
                # Use manual cluster count only when auto-tune is OFF
                clusterer_kwargs['n_clusters'] = n_clusters
                status_text.text(f"🎯 Clustering documents into {n_clusters} clusters...")
            else:
                # Auto-tune will determine optimal cluster count
                status_text.text("🎯 Clustering documents (auto-detecting optimal count)...")
        
        elif algorithm == "dbscan":
            clusterer_kwargs['eps'] = eps
            clusterer_kwargs['min_samples'] = min_samples
        else:  # agglomerative
            clusterer_kwargs['n_clusters'] = n_clusters
        
        clusterer = DocumentClusterer(**clusterer_kwargs)
        cluster_labels = clusterer.fit_predict(embeddings, auto_tune=auto_tune)
        
        st.session_state.cluster_labels = cluster_labels
        st.session_state.clusterer = clusterer
        
        # Check for cancellation
        if st.session_state.processing_cancelled:
            status_text.text("🛑 Processing cancelled after clustering")
            progress_bar.empty()
            st.session_state.processing = False
            return
        
        # Step 4: Generate intelligent cluster names
        status_text.text("🏷️ Generating cluster names...")
        progress_bar.progress(0.90)
        
        # Extract document texts
        document_texts = [result['full_text'] for result in extraction_results]
        
        # Generate names
        cluster_names = clusterer.generate_cluster_names(document_texts, document_names)
        st.session_state.cluster_names = cluster_names
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Reset processing flag
        st.session_state.processing = False
        st.session_state.processing_cancelled = False
        
        st.success(f"🎉 Successfully processed {len(uploaded_file_data)} documents into {len(set(cluster_labels))} clusters!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.processing = False
        st.session_state.processing_cancelled = False


def show_analysis_results():
    """Display analysis results."""
    
    extraction_results = st.session_state.extraction_results
    embeddings = st.session_state.embeddings
    cluster_labels = st.session_state.cluster_labels
    clusterer = st.session_state.clusterer
    document_names = st.session_state.document_names
    cluster_names = st.session_state.cluster_names or {}
    
    # Check if processing was cancelled or incomplete
    if cluster_labels is None or embeddings is None:
        st.warning("⚠️ Processing was cancelled or incomplete. Please process documents again to see full analysis.")
        if extraction_results is not None:
            st.info(f"📄 Text extraction completed for {len(extraction_results)} documents.")
        return
    
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
    
    # Display cluster names
    if cluster_names:
        st.subheader("🏷️ Detected Themes")
        cols = st.columns(min(len(cluster_names), 3))
        for idx, (cluster_id, name) in enumerate(sorted(cluster_names.items())):
            with cols[idx % 3]:
                st.info(f"**Cluster {cluster_id}:** {name}")
    
    st.markdown("---")
    
    # Document details
    st.subheader("📋 Document Details")
    
    df_data = []
    if cluster_labels is not None and extraction_results is not None:
        for i, (result, cluster) in enumerate(zip(extraction_results, cluster_labels)):
            cluster_name = cluster_names.get(cluster, f"Cluster_{cluster}")
            df_data.append({
                'Document': result['filename'],
                'Cluster': cluster_name,
                'Pages': result['page_count'],
                'Words': result['word_count'],
                'Status': '✅' if result['success'] else '❌'
            })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Visualization
    st.subheader("📊 Cluster Visualization")
    
    # Check if we have the necessary data for visualization
    if embeddings is None or cluster_labels is None or document_names is None:
        st.info("⏳ Complete processing to see cluster visualization")
        return
    
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
        # Use use_container_width instead of deprecated width parameter
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 documents for visualization")


def show_organization_options():
    """Show file organization options."""
    
    cluster_labels = st.session_state.cluster_labels
    document_names = st.session_state.document_names
    uploaded_file_data = st.session_state.uploaded_file_data
    cluster_names = st.session_state.cluster_names or {}
    
    # Check if processing was cancelled or incomplete
    if cluster_labels is None or document_names is None:
        st.warning("⚠️ Processing was cancelled or incomplete. Please complete document processing to organize files.")
        return
    
    st.subheader("🗂️ Organization Preview")
    
    # Show clusters with intelligent names
    for cluster_id in sorted(set(cluster_labels)):
        cluster_docs = [doc for doc, label in zip(document_names, cluster_labels) if label == cluster_id]
        cluster_name = cluster_names.get(cluster_id, f"Cluster_{cluster_id}")
        
        with st.expander(f"📁 {cluster_name}"):
            for doc in cluster_docs:
                st.write(f"- {doc}")
    
    st.markdown("---")
    
    # Organization options
    st.subheader("📦 Download Organized Files")
    
    output_format = st.radio(
        "Organization Format",
        ["Single ZIP with folders"],
        help="Download all documents organized in cluster folders"
    )
    
    if st.button("📥 Download Organized Files", type="primary"):
        if uploaded_file_data is None:
            st.error("❌ No uploaded files found. Please process documents first.")
        else:
            try:
                # Create ZIP file in memory
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add files to ZIP organized by cluster
                    for doc_name, cluster_id in zip(document_names, cluster_labels):
                        if doc_name in uploaded_file_data:
                            # Create folder path for this cluster with intelligent name
                            cluster_name = cluster_names.get(cluster_id, f"Cluster_{cluster_id}")
                            folder_name = cluster_name.replace('(', '').replace(')', '').replace(' ', '_')
                            file_path = f"{folder_name}/{doc_name}"
                            
                            # Add file to ZIP
                            zip_file.writestr(file_path, uploaded_file_data[doc_name])
                    
                    # Add a summary report
                    summary = create_organization_summary(document_names, cluster_labels, cluster_names)
                    zip_file.writestr("organization_summary.txt", summary)
                
                # Prepare download
                zip_buffer.seek(0)
                
                st.download_button(
                    label="� Click here to download organized files",
                    data=zip_buffer,
                    file_name="organized_pdfs.zip",
                    mime="application/zip",
                    type="primary"
                )
                
                st.success("✅ ZIP file ready! Click the button above to download.")
                
            except Exception as e:
                st.error(f"❌ Error creating download: {str(e)}")


def create_organization_summary(document_names, cluster_labels, cluster_names=None):
    """Create a text summary of the organization."""
    summary = "=" * 60 + "\n"
    summary += "AI-POWERED PDF ORGANIZATION SUMMARY\n"
    summary += "=" * 60 + "\n\n"
    
    summary += f"Total Documents: {len(document_names)}\n"
    summary += f"Total Clusters: {len(set(cluster_labels))}\n\n"
    
    summary += "=" * 60 + "\n"
    summary += "CLUSTER DETAILS\n"
    summary += "=" * 60 + "\n\n"
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_docs = [doc for doc, label in zip(document_names, cluster_labels) if label == cluster_id]
        
        # Get cluster name
        if cluster_names and cluster_id in cluster_names:
            cluster_name = cluster_names[cluster_id]
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        summary += f"{cluster_name}\n"
        summary += "-" * 40 + "\n"
        
        for doc in cluster_docs:
            summary += f"  • {doc}\n"
        
        summary += "\n"
    
    summary += "=" * 60 + "\n"
    summary += "This organization was created using AI-powered semantic clustering.\n"
    summary += "Documents in the same cluster share similar content and themes.\n"
    summary += "Cluster names are automatically generated based on content analysis.\n"
    summary += "=" * 60 + "\n"
    
    return summary


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
    - **Use multilingual models (🌍) for documents in multiple languages**
    
    ## 🌍 Multilingual Support
    
    **New!** The organizer now supports documents in **50+ languages** including:
    
    - 🇸🇦 **Arabic** (العربية)
    - 🇨🇳 **Chinese** (中文)
    - 🇫🇷 **French** (Français)
    - 🇩🇪 **German** (Deutsch)
    - 🇪🇸 **Spanish** (Español)
    - 🇯🇵 **Japanese** (日本語)
    - 🇷🇺 **Russian** (Русский)
    - 🇮🇹 **Italian** (Italiano)
    - 🇵🇹 **Portuguese** (Português)
    - 🇹🇷 **Turkish** (Türkçe)
    - And many more!
    
    **Recommended Multilingual Models:**
    
    - **paraphrase-multilingual-mpnet-base-v2**: Best quality for mixed-language documents
    - **paraphrase-multilingual-MiniLM-L12-v2**: Faster processing for large multilingual datasets
    - **LaBSE**: Best for cross-lingual matching (109 languages!)
    
    **Example Use Cases:**
    - Mix of Arabic and English research papers
    - International business documents (French, German, English)
    - Multilingual legal documents
    - Academic papers from different countries
    
    ## 📚 Model Information
    
    **all-MiniLM-L6-v2**: Fast, efficient, good for general purposes  
    **all-mpnet-base-v2**: Higher quality, best overall performance  
    **all-distilroberta-v1**: Balanced speed and quality  
    **paraphrase-MiniLM-L6-v2**: Optimized for semantic similarity  
    """)


if __name__ == "__main__":
    main()
