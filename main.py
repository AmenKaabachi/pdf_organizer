"""
Command-line interface for AI-Powered PDF Organizer.

This script provides a CLI for processing and organizing PDF files
using machine learning clustering techniques.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from extraction.pdf_extractor import PDFExtractor
from embeddings.embedding_generator import EmbeddingGenerator
from clustering.document_clusterer import DocumentClusterer
from organization.file_organizer import FileOrganizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI application."""
    parser = argparse.ArgumentParser(
        description="AI-Powered PDF Organizer - Automatically cluster and organize PDF documents"
    )
    
    # Required arguments (except when listing models)
    parser.add_argument(
        "--input-dir",
        help="Directory containing PDF files to organize"
    )
    
    parser.add_argument(
        "--output-dir", 
        help="Directory to save organized files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2", 
        help="Embedding model to use (default: all-MiniLM-L6-v2). Use --list-models to see all options"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--algorithm",
        default="kmeans",
        choices=["kmeans", "dbscan", "agglomerative"],
        help="Clustering algorithm (default: kmeans)"
    )
    
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Number of clusters (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum text length to consider a page valid (default: 100)"
    )
    
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of moving them"
    )
    
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't create timestamped output directory"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PDFs recursively in subdirectories"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle list models request
    if args.list_models:
        print("Available Models:")
        print("=" * 50)
        models = EmbeddingGenerator.list_available_models()
        for name, info in models.items():
            type_icon = "[LOCAL]" if info.get('uses_local', True) else "[API]"
            lang_icon = "[MULTI]" if not info.get('languages', '').startswith('English') else "[EN]"
            size_info = info.get('download_size', f"{info.get('size', 'Unknown')} dim")
            print(f"{type_icon} {lang_icon} {name}")
            print(f"   {info.get('description', 'No description')}")
            print(f"   Size: {size_info}")
            print()
        return
    
    # Validate required arguments for normal operation
    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required (unless using --list-models)")
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        logger.info("🤖 Starting AI-Powered PDF Organizer")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Algorithm: {args.algorithm}")
        
        # Step 1: Extract text from PDFs
        logger.info("📄 Step 1: Extracting text from PDFs...")
        extractor = PDFExtractor(min_text_length=args.min_text_length)
        extraction_results = extractor.extract_from_directory(
            str(input_dir), 
            recursive=args.recursive
        )
        
        # Filter successful extractions
        valid_documents = [doc for doc in extraction_results if doc['success']]
        
        if not valid_documents:
            logger.error("No valid PDF documents found with extractable text")
            sys.exit(1)
        
        logger.info(f"Successfully extracted text from {len(valid_documents)}/{len(extraction_results)} documents")
        
        # Step 2: Generate embeddings
        logger.info("🧠 Step 2: Generating semantic embeddings...")
        # Validate model exists
        available_models = EmbeddingGenerator.list_available_models()
        if args.model not in available_models:
            logger.error(f"Model '{args.model}' not available. Use --list-models to see options.")
            sys.exit(1)
        
        embedding_generator = EmbeddingGenerator(model_name=args.model)
        embeddings, document_names = embedding_generator.generate_document_embeddings(
            valid_documents,
            batch_size=args.batch_size
        )
        
        if len(embeddings) == 0:
            logger.error("No embeddings could be generated")
            sys.exit(1)
        
        logger.info(f"Generated embeddings for {len(embeddings)} documents")
        
        # Step 3: Cluster documents
        logger.info("🎯 Step 3: Clustering documents...")
        clusterer = DocumentClusterer(
            algorithm=args.algorithm,
            n_clusters=args.n_clusters
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        clustering_df = clusterer.create_cluster_summary(document_names, embeddings)
        
        logger.info(f"Clustered documents into {len(set(cluster_labels))} groups")
        logger.info(f"Silhouette Score: {clusterer.silhouette_score:.3f}")
        
        # Step 4: Organize files
        logger.info("🗂️ Step 4: Organizing files...")
        organizer = FileOrganizer(
            base_output_dir=args.output_dir,
            copy_files=args.copy_files,
            create_timestamp_dir=not args.no_timestamp
        )
        
        # Generate cluster names
        cluster_names = organizer.create_cluster_names(clustering_df)
        
        # Organize files
        organization_results = organizer.organize_files(
            clustering_df,
            source_directory=str(input_dir),
            cluster_names=cluster_names
        )
        
        # Print summary
        logger.info("✅ Organization completed successfully!")
        logger.info(f"Total files processed: {organization_results['total_files']}")
        logger.info(f"Successfully organized: {organization_results['successful_operations']}")
        logger.info(f"Failed operations: {organization_results['failed_operations']}")
        
        if 'summary_report_path' in organization_results:
            logger.info(f"Summary report: {organization_results['summary_report_path']}")
        
        # Print cluster breakdown
        print("\n📊 CLUSTER BREAKDOWN:")
        print("=" * 50)
        for cluster_id, stats in organization_results['cluster_statistics'].items():
            print(f"📁 {stats['name']}: {stats['file_count']} files ({stats['percentage']:.1f}%)")
        
        logger.info("🎉 PDF organization completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()