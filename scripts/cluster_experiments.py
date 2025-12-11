"""Clustering experiments harness

Runs a set of clustering algorithms on the test corpus and saves a report
with metrics and cluster contents to debug_output/cluster_experiments_report.json

This script is safe to run locally and uses the same EmbeddingGenerator and
DocumentClusterer implementations from src/.
"""

import json
import os
import sys
from pathlib import Path
from pprint import pprint

# Ensure repo root on sys.path so `src` package imports work when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.pdf_extractor import PDFExtractor
from src.embeddings.embedding_generator import EmbeddingGenerator

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def load_and_embed(model_name='all-mpnet-base-v2'):
    extractor = PDFExtractor()
    files = list(Path('test_pdfs').glob('*.pdf'))
    extraction_results = []
    for f in files:
        res = extractor.extract_text_from_pdf(str(f))
        if res.get('success') and res.get('full_text'):
            extraction_results.append(res)

    eg = EmbeddingGenerator(model_name=model_name, cache_embeddings=False)
    embeddings, doc_names = eg.generate_document_embeddings(extraction_results, batch_size=16, show_progress=False)
    return embeddings, doc_names, extraction_results


def evaluate_clustering(embeddings, labels):
    metrics = {}
    try:
        if len(set(labels)) > 1:
            metrics['silhouette'] = float(silhouette_score(embeddings, labels, metric='cosine'))
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(embeddings, labels))
        else:
            metrics['silhouette'] = -1.0
            metrics['calinski_harabasz'] = 0.0
    except Exception as e:
        metrics['silhouette'] = None
        metrics['calinski_harabasz'] = None
    return metrics


def run_experiments():
    report = {
        'runs': []
    }

    embeddings, doc_names, extraction_results = load_and_embed()
    os.makedirs('debug_output', exist_ok=True)

    # Ensure embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # KMeans grid
    best_km = None
    best_sil = -999
    for k in range(2, min(12, len(embeddings))):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(embeddings)
        metrics = evaluate_clustering(embeddings, labels)
        run = {
            'algorithm': 'kmeans',
            'params': {'n_clusters': k},
            'metrics': metrics,
            'clusters': {}
        }
        for cid in sorted(set(labels)):
            run['clusters'][int(cid)] = [doc for doc, lab in zip(doc_names, labels) if lab == cid]
        report['runs'].append(run)
        if metrics.get('silhouette') is not None and metrics['silhouette'] > best_sil:
            best_sil = metrics['silhouette']
            best_km = run

    # Agglomerative (ward and average)
    for linkage in ['ward', 'average']:
        for k in range(2, min(12, len(embeddings))):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = agg.fit_predict(embeddings)
                metrics = evaluate_clustering(embeddings, labels)
                run = {
                    'algorithm': 'agglomerative',
                    'params': {'n_clusters': k, 'linkage': linkage},
                    'metrics': metrics,
                    'clusters': {}
                }
                for cid in sorted(set(labels)):
                    run['clusters'][int(cid)] = [doc for doc, lab in zip(doc_names, labels) if lab == cid]
                report['runs'].append(run)
            except Exception as e:
                # skip invalid combinations (e.g., ward + cosine weirdness)
                continue

    # DBSCAN sweep on eps
    for eps in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for min_samples in [1, 2, 3]:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = db.fit_predict(embeddings)
            metrics = evaluate_clustering(embeddings, labels)
            run = {
                'algorithm': 'dbscan',
                'params': {'eps': eps, 'min_samples': min_samples},
                'metrics': metrics,
                'clusters': {}
            }
            for cid in sorted(set(labels)):
                run['clusters'][int(cid)] = [doc for doc, lab in zip(doc_names, labels) if lab == cid]
            report['runs'].append(run)

    # Find best run by silhouette (highest)
    best_run = None
    best_sil = -999
    for r in report['runs']:
        sil = r['metrics'].get('silhouette')
        if sil is None:
            continue
        if sil > best_sil:
            best_sil = sil
            best_run = r

    summary = {
        'best_by_silhouette': best_run,
        'best_silhouette': best_sil
    }

    out = {'report': report, 'summary': summary}
    out_file = Path('debug_output/cluster_experiments_report.json')
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved report to {out_file}")

    # Print short summary
    print("Best run by silhouette:")
    pprint(summary['best_by_silhouette'])


if __name__ == '__main__':
    run_experiments()
