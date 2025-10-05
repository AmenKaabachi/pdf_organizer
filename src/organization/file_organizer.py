"""
File organization module for managing and organizing PDF files based on clustering results.

This module provides functionality to create organized directory structures,
move files to appropriate clusters, and generate reports.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import json
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileOrganizer:
    """Organizes files based on clustering results."""
    
    def __init__(self, 
                 base_output_dir: str,
                 create_timestamp_dir: bool = True,
                 copy_files: bool = True):
        """
        Initialize file organizer.
        
        Args:
            base_output_dir: Base directory for organized files
            create_timestamp_dir: Whether to create timestamped subdirectory
            copy_files: Whether to copy files (True) or move them (False)
        """
        self.base_output_dir = Path(base_output_dir)
        self.create_timestamp_dir = create_timestamp_dir
        self.copy_files = copy_files
        
        # Create output directory structure
        self.output_dir = self._create_output_directory()
        
        logger.info(f"File organizer initialized. Output directory: {self.output_dir}")
    
    def organize_files(self, 
                      clustering_results: pd.DataFrame,
                      source_directory: Optional[str] = None,
                      cluster_names: Optional[Dict[int, str]] = None,
                      create_summary: bool = True) -> Dict[str, any]:
        """
        Organize files into clusters based on clustering results.
        
        Args:
            clustering_results: DataFrame with document_name and cluster_id columns
            source_directory: Source directory containing original files
            cluster_names: Optional mapping of cluster IDs to descriptive names
            create_summary: Whether to create organization summary
            
        Returns:
            Dictionary containing organization results and statistics
        """
        logger.info("Starting file organization...")
        
        # Create cluster directories
        cluster_dirs = self._create_cluster_directories(clustering_results, cluster_names)
        
        # Track organization results
        results = {
            'organized_files': [],
            'failed_files': [],
            'cluster_statistics': {},
            'total_files': len(clustering_results),
            'successful_operations': 0,
            'failed_operations': 0
        }
        
        # Organize files by cluster
        for _, row in clustering_results.iterrows():
            try:
                file_result = self._organize_single_file(
                    row, cluster_dirs, source_directory
                )
                
                if file_result['success']:
                    results['organized_files'].append(file_result)
                    results['successful_operations'] += 1
                else:
                    results['failed_files'].append(file_result)
                    results['failed_operations'] += 1
                    
            except Exception as e:
                error_result = {
                    'document_name': row.get('document_name', 'unknown'),
                    'cluster_id': row.get('cluster_id', -1),
                    'success': False,
                    'error': str(e)
                }
                results['failed_files'].append(error_result)
                results['failed_operations'] += 1
                logger.error(f"Error organizing {row.get('document_name')}: {str(e)}")
        
        # Generate cluster statistics
        results['cluster_statistics'] = self._generate_cluster_statistics(
            clustering_results, cluster_names
        )
        
        # Create summary report
        if create_summary:
            summary_path = self._create_organization_summary(results, clustering_results)
            results['summary_report_path'] = str(summary_path)
        
        logger.info(f"Organization completed. {results['successful_operations']}/{results['total_files']} files organized successfully")
        
        return results
    
    def _create_output_directory(self) -> Path:
        """Create the main output directory structure."""
        if self.create_timestamp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.base_output_dir / f"organized_{timestamp}"
        else:
            output_dir = self.base_output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "clusters").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        
        return output_dir
    
    def _create_cluster_directories(self, 
                                  clustering_results: pd.DataFrame,
                                  cluster_names: Optional[Dict[int, str]] = None) -> Dict[int, Path]:
        """Create directories for each cluster."""
        cluster_dirs = {}
        unique_clusters = sorted(clustering_results['cluster_id'].unique())
        
        for cluster_id in unique_clusters:
            if cluster_names and cluster_id in cluster_names:
                dir_name = f"cluster_{cluster_id}_{cluster_names[cluster_id]}"
            else:
                dir_name = f"cluster_{cluster_id}"
            
            # Sanitize directory name
            dir_name = self._sanitize_filename(dir_name)
            
            cluster_dir = self.output_dir / "clusters" / dir_name
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            cluster_dirs[cluster_id] = cluster_dir
            logger.debug(f"Created cluster directory: {cluster_dir}")
        
        return cluster_dirs
    
    def _organize_single_file(self, 
                            row: pd.Series,
                            cluster_dirs: Dict[int, Path],
                            source_directory: Optional[str] = None) -> Dict[str, any]:
        """Organize a single file into its cluster directory."""
        document_name = row['document_name']
        cluster_id = row['cluster_id']
        
        # Find source file
        source_file = self._find_source_file(document_name, source_directory)
        
        if not source_file or not source_file.exists():
            return {
                'document_name': document_name,
                'cluster_id': cluster_id,
                'success': False,
                'error': f'Source file not found: {document_name}',
                'source_path': str(source_file) if source_file else None,
                'destination_path': None
            }
        
        # Determine destination
        cluster_dir = cluster_dirs[cluster_id]
        destination_file = cluster_dir / source_file.name
        
        # Handle name conflicts
        counter = 1
        original_destination = destination_file
        while destination_file.exists():
            stem = original_destination.stem
            suffix = original_destination.suffix
            destination_file = cluster_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            # Copy or move file
            if self.copy_files:
                shutil.copy2(source_file, destination_file)
                operation = "copied"
            else:
                shutil.move(str(source_file), str(destination_file))
                operation = "moved"
            
            return {
                'document_name': document_name,
                'cluster_id': cluster_id,
                'success': True,
                'operation': operation,
                'source_path': str(source_file),
                'destination_path': str(destination_file),
                'file_size_mb': round(destination_file.stat().st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {
                'document_name': document_name,
                'cluster_id': cluster_id,
                'success': False,
                'error': str(e),
                'source_path': str(source_file),
                'destination_path': str(destination_file)
            }
    
    def _find_source_file(self, 
                         document_name: str,
                         source_directory: Optional[str] = None) -> Optional[Path]:
        """Find the source file for a document."""
        if source_directory:
            source_dir = Path(source_directory)
            
            # Try exact match first
            exact_match = source_dir / document_name
            if exact_match.exists():
                return exact_match
            
            # Search recursively
            for pdf_file in source_dir.rglob("*.pdf"):
                if pdf_file.name == document_name:
                    return pdf_file
        
        # If no source directory specified, assume document_name is full path
        potential_path = Path(document_name)
        if potential_path.exists():
            return potential_path
        
        return None
    
    def _generate_cluster_statistics(self, 
                                   clustering_results: pd.DataFrame,
                                   cluster_names: Optional[Dict[int, str]] = None) -> Dict[int, Dict[str, any]]:
        """Generate statistics for each cluster."""
        stats = {}
        
        for cluster_id in sorted(clustering_results['cluster_id'].unique()):
            cluster_data = clustering_results[clustering_results['cluster_id'] == cluster_id]
            
            cluster_stats = {
                'cluster_id': cluster_id,
                'name': cluster_names.get(cluster_id, f'Cluster {cluster_id}') if cluster_names else f'Cluster {cluster_id}',
                'file_count': len(cluster_data),
                'percentage': (len(cluster_data) / len(clustering_results)) * 100,
                'files': cluster_data['document_name'].tolist()
            }
            
            # Add additional metrics if available
            if 'size' in cluster_data.columns:
                cluster_stats['avg_cluster_similarity'] = cluster_data['intra_cluster_similarity'].iloc[0] if 'intra_cluster_similarity' in cluster_data.columns else None
            
            stats[cluster_id] = cluster_stats
        
        return stats
    
    def _create_organization_summary(self, 
                                   results: Dict[str, any],
                                   clustering_results: pd.DataFrame) -> Path:
        """Create comprehensive organization summary report."""
        summary_path = self.output_dir / "reports" / "organization_summary.json"
        
        # Create detailed summary
        summary = {
            'organization_metadata': {
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir),
                'operation_type': 'copy' if self.copy_files else 'move',
                'total_files_processed': results['total_files'],
                'successful_operations': results['successful_operations'],
                'failed_operations': results['failed_operations'],
                'success_rate': (results['successful_operations'] / results['total_files']) * 100 if results['total_files'] > 0 else 0
            },
            'cluster_summary': results['cluster_statistics'],
            'organized_files': results['organized_files'],
            'failed_files': results['failed_files']
        }
        
        # Save JSON summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Create human-readable summary
        readable_summary_path = self.output_dir / "reports" / "organization_summary.txt"
        self._create_readable_summary(summary, readable_summary_path)
        
        # Create CSV summary
        csv_summary_path = self.output_dir / "reports" / "cluster_breakdown.csv"
        clustering_results.to_csv(csv_summary_path, index=False)
        
        logger.info(f"Summary reports created:")
        logger.info(f"  - JSON: {summary_path}")
        logger.info(f"  - Text: {readable_summary_path}")
        logger.info(f"  - CSV: {csv_summary_path}")
        
        return summary_path
    
    def _create_readable_summary(self, summary: Dict[str, any], output_path: Path):
        """Create human-readable text summary."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("📊 PDF ORGANIZATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            metadata = summary['organization_metadata']
            f.write(f"🕐 Generated: {metadata['timestamp']}\n")
            f.write(f"📁 Output Directory: {metadata['output_directory']}\n")
            f.write(f"🔄 Operation: {metadata['operation_type'].title()}\n\n")
            
            # Overall statistics
            f.write("📈 OVERALL STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Files Processed: {metadata['total_files_processed']}\n")
            f.write(f"Successful Operations: {metadata['successful_operations']}\n")
            f.write(f"Failed Operations: {metadata['failed_operations']}\n")
            f.write(f"Success Rate: {metadata['success_rate']:.1f}%\n\n")
            
            # Cluster breakdown
            f.write("🗂️  CLUSTER BREAKDOWN\n")
            f.write("-" * 25 + "\n")
            
            for cluster_id, stats in summary['cluster_summary'].items():
                f.write(f"\n📋 {stats['name']}\n")
                f.write(f"   Files: {stats['file_count']} ({stats['percentage']:.1f}%)\n")
                f.write(f"   Documents:\n")
                for doc in stats['files'][:10]:  # Show first 10
                    f.write(f"     - {doc}\n")
                if len(stats['files']) > 10:
                    f.write(f"     ... and {len(stats['files']) - 10} more\n")
            
            # Failed files (if any)
            if summary['failed_files']:
                f.write(f"\n❌ FAILED OPERATIONS ({len(summary['failed_files'])})\n")
                f.write("-" * 25 + "\n")
                for failed in summary['failed_files']:
                    f.write(f"   - {failed['document_name']}: {failed['error']}\n")
    
    def create_cluster_names(self, 
                           clustering_results: pd.DataFrame,
                           document_texts: Optional[List[str]] = None) -> Dict[int, str]:
        """
        Generate descriptive names for clusters based on content analysis.
        
        Args:
            clustering_results: DataFrame with clustering results
            document_texts: Optional list of document texts for analysis
            
        Returns:
            Dictionary mapping cluster IDs to descriptive names
        """
        cluster_names = {}
        
        # Simple naming based on cluster size and ID
        for cluster_id in sorted(clustering_results['cluster_id'].unique()):
            cluster_data = clustering_results[clustering_results['cluster_id'] == cluster_id]
            size = len(cluster_data)
            
            # Generate name based on patterns in filenames
            filenames = cluster_data['document_name'].tolist()
            common_keywords = self._extract_common_keywords(filenames)
            
            if common_keywords:
                name = f"{common_keywords[0]}_{size}files"
            else:
                name = f"group_{cluster_id}_{size}files"
            
            cluster_names[cluster_id] = self._sanitize_filename(name)
        
        return cluster_names
    
    def _extract_common_keywords(self, filenames: List[str]) -> List[str]:
        """Extract common keywords from filenames."""
        # Simple keyword extraction from filenames
        all_words = []
        
        for filename in filenames:
            # Remove extension and split by common separators
            name_without_ext = Path(filename).stem
            words = name_without_ext.replace('_', ' ').replace('-', ' ').split()
            
            # Filter out common words and numbers
            filtered_words = [
                word.lower() for word in words 
                if len(word) > 2 and not word.isdigit()
                and word.lower() not in ['the', 'and', 'for', 'with', 'pdf', 'doc']
            ]
            
            all_words.extend(filtered_words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:3] if count > 1]
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be filesystem-safe."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove multiple underscores and trim
        filename = '_'.join(filter(None, filename.split('_')))
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:47] + "..."
        
        return filename


def organize_documents(clustering_results: pd.DataFrame,
                      output_directory: str,
                      source_directory: Optional[str] = None,
                      **kwargs) -> Dict[str, any]:
    """
    Convenience function to organize documents based on clustering results.
    
    Args:
        clustering_results: DataFrame with clustering results
        output_directory: Directory to organize files into
        source_directory: Source directory containing original files
        **kwargs: Additional arguments for FileOrganizer
        
    Returns:
        Organization results dictionary
    """
    organizer = FileOrganizer(output_directory, **kwargs)
    return organizer.organize_files(clustering_results, source_directory)