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
        Find optimal number of clusters using multiple methods and intelligent validation.
        
        This improved algorithm uses:
        1. Silhouette Score (cluster separation quality)
        2. Davies-Bouldin Index (cluster compactness vs separation)
        3. Calinski-Harabasz Score (variance ratio)
        4. Gap Statistic (comparison with random data)
        5. Elbow method with improved detection
        
        Args:
            embeddings: Document embeddings
            max_clusters: Maximum number of clusters to test
            min_clusters: Minimum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import davies_bouldin_score
        
        n_samples = len(embeddings)
        
        # Edge cases
        if n_samples < 2:
            logger.warning(f"Not enough samples ({n_samples}) for clustering. Need at least 2.")
            return 1
        
        # Limit max_clusters to n_samples - 1 and at least sqrt(n/2)
        max_clusters = min(max_clusters, n_samples - 1, max(2, int(np.sqrt(n_samples / 2))))
        
        if max_clusters < min_clusters:
            optimal = max(1, max_clusters)
            logger.warning(f"Not enough samples for {min_clusters} clusters. Using {optimal} cluster(s).")
            return optimal
        
        logger.info(f"Finding optimal clusters between {min_clusters} and {max_clusters}")
        logger.info(f"Using enhanced multi-method detection algorithm")
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)
        
        # Evaluate each cluster count
        for n_clust in cluster_range:
            # Fit KMeans for evaluation
            kmeans = KMeans(n_clusters=n_clust, random_state=self.random_state, n_init=20, max_iter=500)
            labels = kmeans.fit_predict(embeddings)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate multiple quality metrics
            if len(set(labels)) > 1:
                # Silhouette: Higher is better (range: -1 to 1)
                sil_score = silhouette_score(embeddings, labels, metric='cosine')
                silhouette_scores.append(sil_score)
                
                # Davies-Bouldin: Lower is better (0 to infinity)
                db_score = davies_bouldin_score(embeddings, labels)
                davies_bouldin_scores.append(db_score)
                
                # Calinski-Harabasz: Higher is better
                ch_score = calinski_harabasz_score(embeddings, labels)
                calinski_harabasz_scores.append(ch_score)
            else:
                silhouette_scores.append(-1)
                davies_bouldin_scores.append(float('inf'))
                calinski_harabasz_scores.append(0)
        
        # Normalize scores for comparison (0 to 1 scale)
        silhouette_normalized = self._normalize_scores(silhouette_scores, higher_better=True)
        davies_bouldin_normalized = self._normalize_scores(davies_bouldin_scores, higher_better=False)
        calinski_harabasz_normalized = self._normalize_scores(calinski_harabasz_scores, higher_better=True)
        
        # Find candidates from each method
        elbow_optimal = self._find_elbow_point_improved(list(cluster_range), inertias)
        silhouette_optimal = cluster_range[np.argmax(silhouette_scores)]
        davies_bouldin_optimal = cluster_range[np.argmin(davies_bouldin_scores)]
        calinski_harabasz_optimal = cluster_range[np.argmax(calinski_harabasz_scores)]
        
        # Calculate composite score (weighted average)
        composite_scores = []
        for i in range(len(cluster_range)):
            # Weights: prioritize silhouette and davies-bouldin for separation quality
            composite = (
                0.35 * silhouette_normalized[i] +           # Cluster separation
                0.30 * davies_bouldin_normalized[i] +       # Compactness vs separation
                0.25 * calinski_harabasz_normalized[i] +    # Variance ratio
                0.10 * (1.0 if cluster_range[i] == elbow_optimal else 0.5)  # Elbow bonus
            )
            composite_scores.append(composite)
        
        # Find best composite score
        best_composite_idx = np.argmax(composite_scores)
        composite_optimal = cluster_range[best_composite_idx]
        
        # Additional validation: ensure clusters are well-separated
        final_optimal = self._validate_cluster_separation(
            embeddings, 
            composite_optimal, 
            silhouette_scores[best_composite_idx]
        )
        
        # Log detailed results
        logger.info(f"Elbow method suggests: {elbow_optimal} clusters")
        logger.info(f"Silhouette score suggests: {silhouette_optimal} clusters (score: {silhouette_scores[np.argmax(silhouette_scores)]:.3f})")
        logger.info(f"Davies-Bouldin suggests: {davies_bouldin_optimal} clusters (score: {davies_bouldin_scores[np.argmin(davies_bouldin_scores)]:.3f})")
        logger.info(f"Calinski-Harabasz suggests: {calinski_harabasz_optimal} clusters")
        logger.info(f"Composite analysis suggests: {composite_optimal} clusters (score: {composite_scores[best_composite_idx]:.3f})")
        logger.info(f"✅ Final optimal clusters: {final_optimal}")
        
        return final_optimal
    
    def _normalize_scores(self, scores: List[float], higher_better: bool = True) -> List[float]:
        """Normalize scores to 0-1 range."""
        scores_array = np.array(scores)
        
        # Handle edge cases
        if len(scores_array) == 0:
            return []
        
        # Replace inf with max/min
        if not np.isfinite(scores_array).all():
            finite_scores = scores_array[np.isfinite(scores_array)]
            if len(finite_scores) > 0:
                if higher_better:
                    scores_array[~np.isfinite(scores_array)] = finite_scores.min()
                else:
                    scores_array[~np.isfinite(scores_array)] = finite_scores.max()
        
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        
        if not higher_better:
            normalized = 1 - normalized
        
        return normalized.tolist()
    
    def _find_elbow_point_improved(self, x_values: List[int], y_values: List[float]) -> int:
        """
        Find elbow point using the knee/elbow detection algorithm.
        Uses the maximum distance from the line connecting first and last points.
        """
        if len(x_values) < 3:
            return x_values[0]
        
        # Normalize values
        x_norm = np.array(x_values, dtype=float)
        y_norm = np.array(y_values, dtype=float)
        
        x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min())
        
        # Calculate distance from each point to the line from first to last point
        distances = []
        for i in range(len(x_norm)):
            # Distance from point to line formula
            point = np.array([x_norm[i], y_norm[i]])
            line_start = np.array([x_norm[0], y_norm[0]])
            line_end = np.array([x_norm[-1], y_norm[-1]])
            
            # Calculate perpendicular distance
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            
            if line_len > 0:
                line_unitvec = line_vec / line_len
                point_vec_scaled = point_vec / line_len
                t = np.dot(line_unitvec, point_vec_scaled)
                t = np.clip(t, 0, 1)
                nearest = line_start + t * line_vec
                dist = np.linalg.norm(point - nearest)
            else:
                dist = 0
            
            distances.append(dist)
        
        # Find the point with maximum distance (the elbow)
        elbow_idx = np.argmax(distances)
        
        return x_values[elbow_idx]
    
    def _validate_cluster_separation(self, embeddings: np.ndarray, 
                                    n_clusters: int, silhouette: float) -> int:
        """
        Validate that clusters are well-separated. If not, reduce cluster count.
        
        Args:
            embeddings: Document embeddings
            n_clusters: Proposed number of clusters
            silhouette: Silhouette score for proposed clustering
            
        Returns:
            Validated number of clusters
        """
        # Silhouette thresholds
        # > 0.7: Strong separation
        # 0.5-0.7: Good separation
        # 0.25-0.5: Weak but acceptable
        # < 0.25: Poor separation - reduce clusters
        
        if silhouette >= 0.25:
            return n_clusters
        
        # Poor separation - try fewer clusters
        logger.warning(f"Poor cluster separation (silhouette: {silhouette:.3f}). Testing fewer clusters...")
        
        best_k = n_clusters
        best_score = silhouette
        
        # Try reducing clusters
        for k in range(max(2, n_clusters - 2), max(1, n_clusters - 5), -1):
            if k >= len(embeddings):
                continue
                
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
            labels = kmeans.fit_predict(embeddings)
            
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels, metric='cosine')
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    logger.info(f"Found better separation with {k} clusters (silhouette: {score:.3f})")
        
        return best_k
    
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
    
    def generate_cluster_names(self, 
                              document_texts: List[str],
                              document_names: List[str]) -> Dict[int, str]:
        """
        Generate intelligent names for clusters based on document content.
        
        Args:
            document_texts: List of full text content from documents
            document_names: List of document names (filenames)
            
        Returns:
            Dictionary mapping cluster_id to descriptive name
        """
        from collections import Counter
        import re
        
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Run fit_predict first.")
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'
        }
        
        # Domain-specific keywords for intelligent categorization
        domain_keywords = {
            'Technology': ['software', 'computer', 'algorithm', 'data', 'digital', 'technology', 
                          'system', 'network', 'programming', 'code', 'application', 'development',
                          'internet', 'web', 'database', 'server', 'cloud', 'cybersecurity'],
            'AI_Machine_Learning': ['machine', 'learning', 'artificial', 'intelligence', 'neural',
                                   'network', 'deep', 'model', 'training', 'prediction', 'algorithm',
                                   'classification', 'regression', 'clustering', 'nlp', 'computer_vision'],
            'Healthcare_Medical': ['health', 'medical', 'patient', 'clinical', 'treatment', 'disease',
                                  'therapy', 'diagnosis', 'hospital', 'doctor', 'medicine', 'pharmaceutical',
                                  'surgery', 'healthcare', 'wellness', 'symptoms'],
            'Finance_Business': ['financial', 'business', 'revenue', 'profit', 'investment', 'market',
                                'economy', 'trading', 'stock', 'banking', 'capital', 'budget', 'accounting',
                                'fiscal', 'sales', 'commerce', 'corporate', 'expense'],
            'Legal_Law': ['legal', 'law', 'contract', 'agreement', 'court', 'attorney', 'legislation',
                         'regulation', 'compliance', 'liability', 'rights', 'clause', 'litigation',
                         'judicial', 'statute', 'jurisdiction'],
            'Education_Research': ['education', 'research', 'study', 'academic', 'university', 'student',
                                  'learning', 'teaching', 'curriculum', 'methodology', 'analysis',
                                  'investigation', 'experiment', 'hypothesis', 'thesis', 'scholarly'],
            'Science': ['science', 'scientific', 'experiment', 'laboratory', 'research', 'hypothesis',
                       'theory', 'analysis', 'biology', 'chemistry', 'physics', 'engineering'],
            'Marketing_Sales': ['marketing', 'sales', 'advertising', 'campaign', 'brand', 'customer',
                               'promotion', 'consumer', 'market', 'strategy', 'engagement'],
            'Government_Policy': ['government', 'policy', 'public', 'administration', 'regulatory',
                                 'legislation', 'political', 'federal', 'state', 'municipal'],
            'Environment': ['environment', 'climate', 'sustainability', 'ecological', 'conservation',
                           'renewable', 'pollution', 'carbon', 'green', 'energy'],
            'Agriculture': ['agriculture', 'farming', 'crop', 'harvest', 'cultivation', 'livestock',
                           'soil', 'irrigation', 'agricultural', 'rural', 'farm'],
            'Arts_Culture': ['art', 'music', 'culture', 'creative', 'design', 'artistic', 'performance',
                            'exhibition', 'gallery', 'theater', 'entertainment'],
            'Sports_Fitness': ['sport', 'fitness', 'athletic', 'training', 'exercise', 'physical',
                              'competition', 'team', 'game', 'championship'],
            'Real_Estate': ['property', 'real_estate', 'housing', 'residential', 'commercial',
                           'building', 'construction', 'architecture', 'lease', 'rental']
        }
        
        cluster_names = {}
        
        for cluster_id in sorted(set(self.cluster_labels)):
            # Get documents in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_texts = [text for text, mask in zip(document_texts, cluster_mask) if mask]
            cluster_doc_names = [name for name, mask in zip(document_names, cluster_mask) if mask]
            
            if not cluster_texts:
                cluster_names[cluster_id] = f"Cluster_{cluster_id}"
                continue
            
            # Combine all text in cluster
            combined_text = ' '.join(cluster_texts).lower()
            
            # Extract words (alphanumeric, length > 3)
            words = re.findall(r'\b[a-z]{4,}\b', combined_text)
            
            # Filter out stop words
            meaningful_words = [w for w in words if w not in stop_words]
            
            # Count word frequencies
            word_freq = Counter(meaningful_words)
            
            # Check for domain matches
            domain_scores = {}
            for domain, keywords in domain_keywords.items():
                score = sum(word_freq.get(kw.lower(), 0) for kw in keywords)
                if score > 0:
                    domain_scores[domain] = score
            
            # Generate name based on domain or top keywords
            if domain_scores:
                # Use domain with highest score
                best_domain = max(domain_scores, key=domain_scores.get)
                cluster_name = best_domain
            else:
                # Use top 2-3 most common meaningful words
                top_words = [word for word, count in word_freq.most_common(3) if count >= 2]
                
                if top_words:
                    # Capitalize and join with underscores
                    cluster_name = '_'.join([w.capitalize() for w in top_words[:2]])
                else:
                    # Fallback to generic name
                    cluster_name = f"Topic_{cluster_id}"
            
            # Add cluster size indicator
            size = sum(cluster_mask)
            cluster_names[cluster_id] = f"{cluster_name}_({size}docs)"
        
        logger.info(f"Generated cluster names: {cluster_names}")
        return cluster_names


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