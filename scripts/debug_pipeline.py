r"""
Quick debugging script to run the pipeline on a directory and print detailed diagnostics.
Run from project root:
    python ./scripts/debug_pipeline.py

It uses a small multilingual model to avoid OOM and disables caching so we force fresh embeddings.
Results are printed and saved to `debug_output/debug_report.json` for inspection.
"""

import json
import sys
import os
from pathlib import Path
import pprint

# Ensure project root is on sys.path so `src` can be imported when running the script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.pdf_extractor import PDFExtractor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.clustering.document_clusterer import DocumentClusterer
from src.organization.file_organizer import FileOrganizer

INPUT_DIR = Path("test_pdfs")
OUTPUT_DIR = Path("debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)

pp = pprint.PrettyPrinter(indent=2)

def run_debug():
    extractor = PDFExtractor(min_text_length=20)
    print(f"Scanning {INPUT_DIR} for PDFs...")
    extraction_results = extractor.extract_from_directory(str(INPUT_DIR), recursive=False)
    valid = [r for r in extraction_results if r.get('success')]
    print(f"Extracted {len(valid)} successful documents (+{len(extraction_results)-len(valid)} failed)")

    # show filenames and text lengths
    docs_meta = []
    for r in valid:
        docs_meta.append({
            'filename': r['filename'],
            'chars': len(r.get('full_text','')), 
            'words': len(r.get('full_text','').split()),
            'sample': r.get('full_text','')[:300].replace('\n',' ') if r.get('full_text') else ''
        })
    pp.pprint(docs_meta)

    # Use a small multilingual model to avoid resource issues
    # Use a higher-quality embedding model for clustering
    model_name = 'all-mpnet-base-v2'
    print(f"\nLoading EmbeddingGenerator (no cache) with model: {model_name}")
    eg = EmbeddingGenerator(model_name=model_name, cache_embeddings=False)

    texts = [r['full_text'] for r in valid]
    names = [r['filename'] for r in valid]

    print(f"Generating embeddings for {len(texts)} docs (this may take a moment)...")
    embeddings, document_names = eg.generate_document_embeddings(valid, batch_size=8, show_progress=False)
    print(f"Embeddings shape: {getattr(embeddings, 'shape', len(embeddings))}")

    # Cluster
    print("\nClustering with DocumentClusterer (auto-tune)...")
    clusterer = DocumentClusterer(n_clusters=None)
    labels = clusterer.fit_predict(embeddings)
    print(f"Found clusters: {sorted(set(labels))}")

    summary_df = clusterer.create_cluster_summary(document_names, embeddings)

    # Generate names using organizer
    organizer = FileOrganizer(str(OUTPUT_DIR), create_timestamp_dir=False, copy_files=True)
    document_texts_map = {r['filename']: r['full_text'] for r in valid}
    # Build document_embeddings mapping (filename -> embedding) when available
    doc_emb_map = {}
    try:
        if embeddings is not None and len(embeddings) == len(document_names):
            for name, emb in zip(document_names, embeddings):
                doc_emb_map[name] = emb
    except Exception:
        doc_emb_map = {}

    cluster_names = organizer.create_cluster_names(summary_df, document_texts=document_texts_map, document_embeddings=doc_emb_map)

    print("\nCluster names mapping:")
    pp.pprint(cluster_names)

    # Print cluster contents
    clusters = {}
    for row in summary_df.to_dict(orient='records'):
        cid = int(row['cluster_id'])
        clusters.setdefault(cid, []).append(row['document_name'])

    print("\nCluster contents:")
    for cid in sorted(clusters.keys()):
        print(f"Cluster {cid} ({len(clusters[cid])} docs):")
        for fn in clusters[cid]:
            txt = document_texts_map.get(fn,'')
            print(f"  - {fn} | chars={len(txt)} | sample={txt[:200].replace('\n',' ')}")

    # Save debug report
    report = {
        'model': model_name,
        'documents': docs_meta,
        'cluster_names': cluster_names,
        'clusters': clusters,
        'clusterer_info': clusterer.get_cluster_info()
    }

    out_file = OUTPUT_DIR / 'debug_report.json'
    # Make JSON-safe: convert numpy keys/types to native Python types
    def make_json_safe(o):
        try:
            import numpy as _np
        except Exception:
            _np = None

        if isinstance(o, dict):
            new = {}
            for k, v in o.items():
                # keys must be strings for JSON
                if isinstance(k, (int,)):
                    key = str(k)
                else:
                    try:
                        # numpy ints
                        if _np is not None and isinstance(k, _np.generic):
                            key = str(int(k))
                        else:
                            key = str(k)
                    except Exception:
                        key = str(k)
                new[key] = make_json_safe(v)
            return new
        elif isinstance(o, list):
            return [make_json_safe(i) for i in o]
        else:
            # numpy scalar handling
            if _np is not None:
                if isinstance(o, _np.integer):
                    return int(o)
                if isinstance(o, _np.floating):
                    return float(o)
                if isinstance(o, _np.ndarray):
                    return o.tolist()
            return o

    safe_report = make_json_safe(report)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(safe_report, f, ensure_ascii=False, indent=2)

    print(f"\nDebug report saved to: {out_file}")

if __name__ == '__main__':
    run_debug()
