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
import re
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from typing import Any
import numpy as np

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
                           document_texts: Optional[Dict[str, str]] = None,
                           document_embeddings: Optional[Dict[str, Any]] = None) -> Dict[int, str]:
        """
        Generate descriptive names for clusters based on content analysis.
        
        Args:
            clustering_results: DataFrame with clustering results
            document_texts: Optional list of document texts for analysis
            
        Returns:
            Dictionary mapping cluster IDs to descriptive names
        """
        cluster_names = {}
        used_names = set()

        # Prepare stop words (combine English + simple French list)
    # Prepare stop words: always have a French list; attempt to combine with sklearn English stop words
        french_stop_words = {
            'le','la','les','de','du','des','un','une','et','en','dans','pour','par','sur','avec',
            'pas','plus','au','aux','ce','ces','qui','que','quoi','dont','où','comme','entre','sans',
            'sous','chez','il','elle','nous','vous','ils','elles','être','avoir','faire','été','son',
            'sa','ses','se','leur','leurs','mon','ton','notre','votre','mais','ou','si','non','donc',
            'or','ni','car','ça',
            # common French auxiliaries / short verbs that were polluting names
            'est','sont','ont','a','ai','as','avons','avez','ont','étaient','être','été'
        }
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            combined_stop_words = set(ENGLISH_STOP_WORDS).union(french_stop_words)
        except Exception:
            # If sklearn isn't available at runtime in this environment, fall back to French-only
            combined_stop_words = set(french_stop_words)

        # Lazy spaCy loader for optional French lemmatization/normalization
        SPACY_NLP: Any = None
        def _load_spacy():
            nonlocal SPACY_NLP
            if SPACY_NLP is not None:
                return SPACY_NLP
            try:
                import spacy
                try:
                    SPACY_NLP = spacy.load("fr_core_news_sm")
                except Exception:
                    # fallback to blank French model if the small pipeline isn't installed
                    SPACY_NLP = spacy.blank("fr")
                return SPACY_NLP
            except Exception:
                SPACY_NLP = None
                return None

        def _lemmatize_phrase(phrase: str) -> str:
            """Lemmatize a short phrase using spaCy French model if available."""
            nlp = _load_spacy()
            if not nlp:
                return phrase
            try:
                doc = nlp(phrase.replace('_', ' '))
                lemmas = [tok.lemma_.lower() for tok in doc if not (tok.is_punct or tok.is_space)]
                # remove stop words and short tokens
                lemmas = [l for l in lemmas if len(l) > 1 and l not in combined_stop_words]
                return '_'.join(lemmas)
            except Exception:
                return phrase

        # Additional tokens to ignore in generated names
        noisy_tokens = set([
            'unknown', 'uncertain', 'misc', 'mixed', 'document', 'documents', 'pdf', 'report', 'general', 'diverse',
            # common file/metadata words
            'cours', 'chapitre', 'introduction', 'exercice', 'exercices', 'corrig', 'corrige', 'corrigés', 'corrigés',
            'table', 'matières', 'matieres', 'sommaire', 'annales', 'étude', 'studyrama', 'cours', 'chapitre'
        ])

        # Helper to clean and normalize keywords
        def _normalize_keyword(k: str) -> str:
            k = k.strip()
            # Separate camel case like 'ComplexesEXOSCORRIGES' -> 'Complexes EXOSCORRIGES'
            try:
                k = re.sub(r'([a-z])([A-Z])', r'\1 \2', k)
            except Exception:
                pass
            k = k.lower()
            for ch in "<>:\"/\\|?*(),.;:'`":
                k = k.replace(ch, ' ')
            k = '_'.join(filter(None, k.split()))
            return k

        # Generate names using document texts when available, fallback to filenames
        for cluster_id in sorted(clustering_results['cluster_id'].unique()):
            cluster_data = clustering_results[clustering_results['cluster_id'] == cluster_id]
            size = len(cluster_data)

            filenames = cluster_data['document_name'].tolist()

            name = None
            keywords = []

            # Try embedding-weighted TF-IDF naming if embeddings are provided
            if document_embeddings:
                try:
                    # Collect texts and embeddings for docs in this cluster
                    texts_map = {}
                    emb_list = []
                    names_with_emb = []
                    for fn in filenames:
                        emb = document_embeddings.get(fn)
                        txt = (document_texts.get(fn) if document_texts else '') or ''
                        if emb is not None:
                            names_with_emb.append(fn)
                            emb_list.append(np.asarray(emb))
                            texts_map[fn] = txt

                    if emb_list:
                        # Compute centroid and per-document similarity weights
                        emb_matrix = np.vstack(emb_list)
                        # normalize rows
                        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        emb_normed = emb_matrix / norms
                        centroid = np.mean(emb_normed, axis=0)
                        # cosine similarities
                        centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
                        sims = (emb_normed @ centroid_norm)

                        # Select top-N representative documents closest to centroid
                        try:
                            import numpy as _np
                            top_n = min(3, len(names_with_emb))
                            top_idx = _np.argsort(sims)[-top_n:][::-1]
                            top_docs = [names_with_emb[i] for i in top_idx]
                            top_texts = [texts_map.get(n, '') or Path(n).stem.replace('_', ' ') for n in top_docs]
                        except Exception:
                            top_docs = names_with_emb[:3]
                            top_texts = [texts_map.get(n, '') or Path(n).stem.replace('_', ' ') for n in top_docs]

                        # Curated blacklist of noisy French tokens/phrases to drop
                        curated_blacklist = set([
                            'alors', 'soit', 'donne', 'donnees', 'données', 'exercice', 'exercices',
                            'corrige', 'corriges', 'corrigés', 'page', 'chapitre', 'introduction',
                            'table', 'matiere', 'matières', 'sommaire', 'annales', 'cours', 'document',
                            'pdf', 'figure', 'tableau', 'section'
                        ])

                        # 1) Try YAKE on the top documents (prefer multi-word phrases)
                        keywords = []
                        try:
                            import yake
                            joined = '\n'.join([t for t in top_texts if t and len(t) > 20])
                            if joined:
                                kw_extractor = yake.KeywordExtractor(lan='fr', n=3, dedupLim=0.8, top=12)
                                raw_kws = kw_extractor.extract_keywords(joined)
                                # prefer multiword phrases first
                                kws = [kw for kw, score in raw_kws if ' ' in kw]
                                kws += [kw for kw, score in raw_kws if ' ' not in kw]
                                keywords = kws
                        except Exception:
                            keywords = []

                        # 2) TF-IDF fallback on the top documents with emphasis on ngrams (2-3)
                        if not keywords:
                            try:
                                from sklearn.feature_extraction.text import TfidfVectorizer
                                sw = list(combined_stop_words) if combined_stop_words is not None else None
                                vect = TfidfVectorizer(max_features=300, stop_words=sw, ngram_range=(2, 3), min_df=1)
                                docs_for_tfidf = [t if t and len(t) > 20 else Path(n).stem.replace('_', ' ') for n, t in zip(top_docs, top_texts)]
                                X = vect.fit_transform(docs_for_tfidf)
                                feature_names = vect.get_feature_names_out()
                                scores = X.sum(axis=0).A1
                                idx = _np.argsort(scores)[-15:][::-1]
                                keywords = [feature_names[i] for i in idx if scores[i] > 0]
                            except Exception:
                                keywords = []

                        # If still nothing, use filename stems of top docs
                        if not keywords:
                            keywords = [Path(n).stem.replace('_', ' ') for n in top_docs]

                        # Filter keywords: remove blacklist and noisy tokens, require alphabetic content
                        filtered = []
                        for k in keywords:
                            kk = k.strip().lower()
                            # remove tokens consisting solely of short function words
                            parts = [p for p in kk.split() if len(p) > 1]
                            if not parts:
                                continue
                            if any(p in curated_blacklist for p in parts):
                                continue
                            # remove if it's mostly numeric or contains few letters
                            alpha_ratio = sum(c.isalpha() for c in kk) / max(1, len(kk))
                            if alpha_ratio < 0.5:
                                continue
                            filtered.append(kk)

                        # Optionally filter by POS (prefer nouns) if spaCy FR is available
                        try:
                            nlp = _load_spacy()
                            if nlp and filtered:
                                nouny = []
                                for ph in filtered:
                                    doc_ph = nlp(ph)
                                    # keep phrase if contains at least one noun or proper noun
                                    if any(tok.pos_ in ('NOUN', 'PROPN') for tok in doc_ph):
                                        nouny.append(ph)
                                if nouny:
                                    filtered = nouny
                        except Exception:
                            pass

                        # Semantic reranking: encode candidate phrases and pick those closest to centroid
                        try:
                            # Only attempt if centroid exists in this scope and we have candidates
                            if filtered and 'centroid' in locals():
                                try:
                                    from sentence_transformers import SentenceTransformer
                                    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
                                    cand_embs = model.encode(filtered, convert_to_numpy=True, show_progress_bar=False)
                                    # normalize
                                    emb_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
                                    emb_norms[emb_norms == 0] = 1.0
                                    cand_norm = cand_embs / emb_norms
                                    cent_norm = centroid / (np.linalg.norm(centroid) or 1.0)
                                    sims_ph = (cand_norm @ cent_norm).ravel()
                                    order = sims_ph.argsort()[::-1]
                                    semantic_selected = []
                                    for i in order:
                                        ph = filtered[int(i)]
                                        # skip blacklist-like tokens
                                        parts = [p for p in ph.split() if len(p) > 1]
                                        if any(p in curated_blacklist for p in parts):
                                            continue
                                        semantic_selected.append(ph)
                                        if len(semantic_selected) >= 2:
                                            break
                                    if semantic_selected:
                                        keywords = semantic_selected
                                    else:
                                        keywords = filtered
                                except Exception:
                                    keywords = filtered
                            else:
                                keywords = filtered
                        except Exception:
                            keywords = filtered

                    else:
                        keywords = []
                except Exception:
                    keywords = []

            # Try using document_texts mapping if provided (original pipeline) only if embedding-based keywords not found
            if document_texts and not keywords:
                texts = [document_texts.get(fn, '') or '' for fn in filenames]
                texts = [t for t in texts if t and len(t.strip()) > 20]

                if texts:
                    # Language detection: simple heuristic based on French stopwords ratio
                    try:
                        total_tokens = 0
                        french_tokens = 0
                        for t in texts:
                            toks = [w.lower() for w in t.split() if len(w) > 1]
                            total_tokens += len(toks)
                            french_tokens += sum(1 for w in toks if w in french_stop_words)
                        lang = 'fr' if total_tokens > 0 and (french_tokens / total_tokens) > 0.03 else 'en'
                    except Exception:
                        lang = 'fr'

                    # 1) Try YAKE (preferred) for short keyphrases
                    keywords = []
                    try:
                        import yake
                        joined = '\n'.join(texts)
                        kw_extractor = yake.KeywordExtractor(lan=lang, n=3, dedupLim=0.9, top=8)
                        raw_kws = kw_extractor.extract_keywords(joined)
                        keywords = [kw for kw, score in raw_kws]
                        keywords = [k for k in keywords if len(k.strip()) > 1]
                    except Exception:
                        keywords = []

                    # 2) TF-IDF fallback for multi-word phrases
                    if not keywords:
                        try:
                            from sklearn.feature_extraction.text import TfidfVectorizer
                            sw = list(combined_stop_words) if combined_stop_words is not None else None
                            vect = TfidfVectorizer(max_features=200, stop_words=sw, ngram_range=(1, 3))
                            X = vect.fit_transform(texts)
                            feature_names = vect.get_feature_names_out()
                            import numpy as _np
                            scores = _np.asarray(X.mean(axis=0)).ravel()
                            top_idx = scores.argsort()[-15:][::-1]
                            keywords = [feature_names[i] for i in top_idx if len(feature_names[i]) > 2]
                        except Exception:
                            keywords = []

                    # Clean and filter keywords/phrases
                    cleaned = []
                    for k in keywords:
                        nk = _normalize_keyword(k)
                        if not nk:
                            continue
                        # drop purely numeric or too short tokens
                        if nk.isdigit() or len(nk) <= 2:
                            continue
                        # drop noisy tokens and common metadata words
                        if any(tok in nk for tok in noisy_tokens):
                            continue
                        # drop stopword-only phrases
                        parts = nk.split('_')
                        if all(p in combined_stop_words for p in parts if p):
                            continue
                        cleaned.append(nk)

                    # Optionally lemmatize phrases (improves French forms)
                    lemmatized = []
                    for term in cleaned:
                        try:
                            lem = _lemmatize_phrase(term)
                            if lem and len(lem) > 1:
                                lemmatized.append(lem)
                        except Exception:
                            lemmatized.append(term)

                    candidates = lemmatized or cleaned

                    if candidates:
                        # Deduplicate substrings and keep 1-3 representative phrases
                        top_terms = []
                        for term in candidates:
                            if not any((term in t or t in term) for t in top_terms):
                                top_terms.append(term)
                            if len(top_terms) >= 3:
                                break

                        # Format terms: Title case, replace underscores with spaces then underscore for filename
                        def fmt(tok: str) -> str:
                            return tok.replace('_', ' ').title().replace(' ', '_')

                        short = '_'.join([fmt(t) for t in top_terms[:3]])

                        # If cluster seems mixed (many different topics), use 'Mixte' prefix, else no prefix
                        # Simple heuristic: if we have fewer candidates than docs, it's probably mixed
                        if len(top_terms) < max(1, min(3, size // 2)):
                            prefix = 'Mixte_'
                        else:
                            prefix = ''

                        suffix = f"{size}doc" if size == 1 else f"{size}docs"
                        name = f"{prefix}{short}_{suffix}"

            # Fallback to filename-based naming
            if not name:
                common_keywords = self._extract_common_keywords(filenames)
                chosen_key = None
                if common_keywords:
                    for cand in common_keywords:
                        nk = _normalize_keyword(cand)
                        if not nk:
                            continue
                        if nk in noisy_tokens:
                            continue
                        # avoid curated generic words
                        if 'cours' in nk or 'document' in nk or 'pdf' in nk:
                            continue
                        if len(nk) > 3:
                            chosen_key = nk
                            break

                if chosen_key:
                    name = f"{chosen_key.capitalize()}_{size}files"
                else:
                    # Use the first filename stem cleaned as a last-resort readable name
                    first_stem = _normalize_keyword(Path(filenames[0]).stem) if filenames else f"cluster{cluster_id}"
                    name = f"{first_stem}_{size}files"

            # Final cleaning and uniqueness
            name = self._sanitize_filename(name)
            base_name = name
            counter = 1
            while name in used_names:
                name = f"{base_name}_{counter}"
                counter += 1

            used_names.add(name)
            cluster_names[cluster_id] = name

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

        # Return most common words (include singletons to avoid falling back to generic names)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:3]]
    
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