"""
Document clustering using various machine learning algorithms.

This module provides functionality to cluster documents based on their embeddings,
with support for different clustering algorithms and automatic parameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentClusterer:
    """Clusters documents based on their semantic embeddings."""
    
    CLUSTERING_ALGORITHMS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering
    }
    
    def __init__(self, 
                 algorithm: str = 'kmeans',
                 n_clusters: Optional[int] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize document clusterer.
        
        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'agglomerative')
            n_clusters: Number of clusters (None for auto-detection)
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for clustering algorithm
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.silhouette_score = None
        self.calinski_harabasz_score = None
        
        # Validate algorithm
        if algorithm not in self.CLUSTERING_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Choose from: {list(self.CLUSTERING_ALGORITHMS.keys())}")
    
    def fit_predict(self, 
                    embeddings: np.ndarray,
                    auto_tune: bool = True) -> np.ndarray:
        """
        Fit clustering model and predict cluster labels.
        
        Args:
            embeddings: Document embeddings array
            auto_tune: Whether to automatically tune parameters
            
        Returns:
            Array of cluster labels
        """
        if embeddings.size == 0:
            logger.warning("Empty embeddings array provided")
            return np.array([])
        
        n_samples = len(embeddings)
        
        # Handle edge case: only 1 document
        if n_samples == 1:
            logger.warning("Only 1 document provided. Assigning to single cluster.")
            self.cluster_labels = np.array([0])
            self.silhouette_score = None
            self.calinski_harabasz_score = None
            return self.cluster_labels
        
        logger.info(f"Clustering {n_samples} documents using {self.algorithm}")
        
        # Auto-tune number of clusters if needed
        if self.n_clusters is None and auto_tune:
            self.n_clusters = self._find_optimal_clusters(embeddings)
        
        # Ensure n_clusters doesn't exceed n_samples
        if self.n_clusters and self.n_clusters >= n_samples:
            logger.warning(f"n_clusters ({self.n_clusters}) >= n_samples ({n_samples}). "
                         f"Setting n_clusters to {max(1, n_samples - 1)}")
            self.n_clusters = max(1, n_samples - 1)
        
        # Initialize clusterer
        self._initialize_clusterer()
        
        # Fit and predict
        try:
            if self.algorithm in ['kmeans', 'agglomerative']:
                self.cluster_labels = self.clusterer.fit_predict(embeddings)
            else:  # DBSCAN
                self.cluster_labels = self.clusterer.fit_predict(embeddings)
            
            # Calculate metrics
            self._calculate_metrics(embeddings)
            
            # Extract cluster centers if available
            if hasattr(self.clusterer, 'cluster_centers_'):
                self.cluster_centers = self.clusterer.cluster_centers_
            
            logger.info(f"Clustering completed. Found {len(set(self.cluster_labels))} clusters")
            return self.cluster_labels
            
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            raise
    
    def _initialize_clusterer(self):
        """Initialize the clustering algorithm."""
        algorithm_class = self.CLUSTERING_ALGORITHMS[self.algorithm]
        
        if self.algorithm == 'kmeans':
            self.clusterer = algorithm_class(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                **self.kwargs
            )
        elif self.algorithm == 'dbscan':
            self.clusterer = algorithm_class(
                eps=self.kwargs.get('eps', 0.5),
                min_samples=self.kwargs.get('min_samples', 5),
                **{k: v for k, v in self.kwargs.items() if k not in ['eps', 'min_samples']}
            )
        elif self.algorithm == 'agglomerative':
            self.clusterer = algorithm_class(
                n_clusters=self.n_clusters,
                linkage=self.kwargs.get('linkage', 'ward'),
                **{k: v for k, v in self.kwargs.items() if k != 'linkage'}
            )
    
    def _find_optimal_clusters(self, 
                             embeddings: np.ndarray,
                             max_clusters: int = 10,
                             min_clusters: int = 2) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            embeddings: Document embeddings
            max_clusters: Maximum number of clusters to test
            min_clusters: Minimum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        n_samples = len(embeddings)
        
        # Edge cases
        if n_samples < 2:
            logger.warning(f"Not enough samples ({n_samples}) for clustering. Need at least 2.")
            return 1
        
        # Limit max_clusters to n_samples - 1
        max_clusters = min(max_clusters, n_samples - 1)
        
        if max_clusters < min_clusters:
            optimal = max(1, max_clusters)
            logger.warning(f"Not enough samples for {min_clusters} clusters. Using {optimal} cluster(s).")
            return optimal
        
        logger.info(f"Finding optimal clusters between {min_clusters} and {max_clusters}")
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)
        
        for n_clust in cluster_range:
            # Fit KMeans for evaluation
            kmeans = KMeans(n_clusters=n_clust, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if len(set(labels)) > 1:
                sil_score = silhouette_score(embeddings, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow point
        optimal_clusters = self._find_elbow_point(list(cluster_range), inertias)
        
        # Validate with silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        silhouette_optimal = cluster_range[best_silhouette_idx]
        
        logger.info(f"Elbow method suggests: {optimal_clusters} clusters")
        logger.info(f"Silhouette score suggests: {silhouette_optimal} clusters")
        
        # Choose the one with better silhouette score if close
        if abs(optimal_clusters - silhouette_optimal) <= 1:
            return silhouette_optimal
        else:
            return optimal_clusters
    
    def _find_elbow_point(self, x_values: List[int], y_values: List[float]) -> int:
        """Find elbow point in the curve using the elbow method."""
        if len(x_values) < 3:
            return x_values[0]
        
        # Calculate differences
        differences = []
        for i in range(1, len(y_values)):
            diff = y_values[i-1] - y_values[i]
            differences.append(diff)
        
        # Find the point where the difference starts to level off
        max_diff_idx = 0
        max_relative_diff = 0
        
        for i in range(1, len(differences)):
            if differences[i-1] > 0:  # Avoid division by zero
                relative_diff = differences[i] / differences[i-1]
                if relative_diff < max_relative_diff or max_relative_diff == 0:
                    max_relative_diff = relative_diff
                    max_diff_idx = i
        
        return x_values[max_diff_idx + 1]
    
    def _calculate_metrics(self, embeddings: np.ndarray):
        """Calculate clustering quality metrics."""
        try:
            if len(set(self.cluster_labels)) > 1:
                self.silhouette_score = silhouette_score(embeddings, self.cluster_labels)
                self.calinski_harabasz_score = calinski_harabasz_score(embeddings, self.cluster_labels)
            else:
                self.silhouette_score = 0
                self.calinski_harabasz_score = 0
                
            logger.info(f"Silhouette Score: {self.silhouette_score:.3f}")
            logger.info(f"Calinski-Harabasz Score: {self.calinski_harabasz_score:.3f}")
            
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {str(e)}")
            self.silhouette_score = 0
            self.calinski_harabasz_score = 0
    
    def create_cluster_summary(self, 
                             document_names: List[str],
                             embeddings: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create a summary DataFrame of clustering results.
        
        Args:
            document_names: List of document names/identifiers
            embeddings: Optional embeddings for calculating cluster statistics
            
        Returns:
            DataFrame with clustering results
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Run fit_predict first.")
        
        # Create base DataFrame
        df = pd.DataFrame({
            'document_name': document_names[:len(self.cluster_labels)],
            'cluster_id': self.cluster_labels
        })
        
        # Add cluster statistics
        cluster_stats = []
        for cluster_id in sorted(set(self.cluster_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            cluster_info = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': (cluster_size / len(self.cluster_labels)) * 100
            }
            
            # Add embedding statistics if available
            if embeddings is not None:
                cluster_embeddings = embeddings[cluster_mask]
                cluster_info.update({
                    'avg_distance_to_center': self._calculate_avg_distance_to_center(
                        cluster_embeddings, cluster_id),
                    'intra_cluster_similarity': self._calculate_intra_cluster_similarity(
                        cluster_embeddings)
                })
            
            cluster_stats.append(cluster_info)
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        
        # Merge with main DataFrame
        result_df = df.merge(cluster_stats_df, on='cluster_id', how='left')
        
        return result_df.sort_values(['cluster_id', 'document_name'])
    
    def _calculate_avg_distance_to_center(self, 
                                        cluster_embeddings: np.ndarray,
                                        cluster_id: int) -> float:
        """Calculate average distance to cluster center."""
        try:
            if self.cluster_centers is not None and cluster_id < len(self.cluster_centers):
                center = self.cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                return float(np.mean(distances))
            else:
                # Calculate centroid manually
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                return float(np.mean(distances))
        except:
            return 0.0
    
    def _calculate_intra_cluster_similarity(self, cluster_embeddings: np.ndarray) -> float:
        """Calculate average intra-cluster similarity."""
        try:
            if len(cluster_embeddings) < 2:
                return 1.0
            
            # Calculate pairwise similarities
            similarities = np.dot(cluster_embeddings, cluster_embeddings.T)
            
            # Get upper triangle (excluding diagonal)
            n = len(similarities)
            upper_triangle_indices = np.triu_indices(n, k=1)
            similarities_upper = similarities[upper_triangle_indices]
            
            return float(np.mean(similarities_upper))
        except:
            return 0.0
    
    def visualize_clusters(self, 
                          embeddings: np.ndarray,
                          document_names: List[str],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 2D visualization of clusters using PCA.
        
        Args:
            embeddings: Document embeddings
            document_names: Document names for labeling
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Run fit_predict first.")
        
        n_samples = len(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Handle edge cases for small datasets
        if n_samples == 1:
            # Single document - show as a single point
            ax.scatter([0], [0], c='blue', s=200, alpha=0.7, label='Cluster 0')
            ax.set_xlabel('Document (Single Point)')
            ax.set_ylabel('No Dimensionality Reduction Needed')
            ax.set_title('Single Document Visualization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(0, -0.1, document_names[0], ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        elif n_samples == 2:
            # Two documents - plot in 1D (line)
            embeddings_2d = np.column_stack([np.arange(n_samples), np.zeros(n_samples)])
            
            unique_labels = sorted(set(self.cluster_labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = self.cluster_labels == label
                ax.scatter(embeddings_2d[mask, 0], 
                          embeddings_2d[mask, 1],
                          c=[colors[i]], 
                          label=f'Cluster {label}',
                          alpha=0.7,
                          s=200)
            
            ax.set_xlabel('Document Index')
            ax.set_ylabel('Cluster Axis')
            ax.set_title('Two Documents Visualization (Linear)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Standard case: 3+ documents - use PCA
            # Determine number of components based on samples
            n_components = min(2, n_samples, embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=self.random_state)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # If PCA only returned 1 component, add a zero column
            if embeddings_2d.shape[1] == 1:
                embeddings_2d = np.column_stack([embeddings_2d, np.zeros(n_samples)])
            
            # Plot points colored by cluster
            unique_labels = sorted(set(self.cluster_labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = self.cluster_labels == label
                ax.scatter(embeddings_2d[mask, 0], 
                          embeddings_2d[mask, 1],
                          c=[colors[i]], 
                          label=f'Cluster {label}',
                          alpha=0.7,
                          s=50)
            
            # Add cluster centers if available
            if self.cluster_centers is not None and len(embeddings) > 2:
                centers_2d = pca.transform(self.cluster_centers)
                if centers_2d.shape[1] == 1:
                    centers_2d = np.column_stack([centers_2d, np.zeros(len(self.cluster_centers))])
                ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                          c='black', marker='x', s=200, linewidths=3,
                          label='Centroids')
            
            # Set labels with variance explanation
            if n_components == 2:
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            else:
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax.set_ylabel('Auxiliary Axis')
            
            ax.set_title(f'Document Clusters Visualization ({self.algorithm.title()})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to: {save_path}")
        
        return fig
    
    def get_cluster_info(self) -> Dict[str, any]:
        """Get comprehensive clustering information."""
        if self.cluster_labels is None:
            return {}
        
        unique_labels = sorted(set(self.cluster_labels))
        
        return {
            'algorithm': self.algorithm,
            'n_clusters': len(unique_labels),
            'cluster_labels': unique_labels,
            'n_documents': len(self.cluster_labels),
            'silhouette_score': self.silhouette_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'cluster_sizes': [np.sum(self.cluster_labels == label) for label in unique_labels]
        }


def cluster_documents(embeddings: np.ndarray,
                     algorithm: str = 'kmeans',
                     n_clusters: Optional[int] = None,
                     **kwargs) -> Tuple[np.ndarray, DocumentClusterer]:
    """
    Convenience function to cluster documents.
    
    Args:
        embeddings: Document embeddings
        algorithm: Clustering algorithm to use
        n_clusters: Number of clusters (None for auto-detection)
        **kwargs: Additional arguments for clusterer
        
    Returns:
        Tuple of (cluster_labels, fitted_clusterer)
    """
    clusterer = DocumentClusterer(
        algorithm=algorithm,
        n_clusters=n_clusters,
        **kwargs
    )
    
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer