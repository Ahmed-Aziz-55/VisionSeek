

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from app.services.searcher import ImageSearcher
from app.evaluation.metrics import RetrievalEvaluator
from app.evaluation.duplicate_detector import DuplicateDetector

# ---- Config ----
K_VALUES = [1, 5, 10]
EVAL_SAMPLE_SIZE = None      # subset for recall/precision — full 31K would mean 31K live CLIP+FAISS calls
LATENCY_SAMPLE_SIZE = 100
DUPLICATE_THRESHOLD = 0.98

MAPPING_PATH = "embeddings/image_mapping.json"
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
REPORT_DIR = Path("app/evaluation/reports")


def main():
    print("Loading searcher (FAISS index + CLIP model)...")
    searcher = ImageSearcher()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    print(f"Loaded mapping with {len(mapping)} records\n")

    print(f"Running retrieval evaluation on {EVAL_SAMPLE_SIZE} sampled queries...")
    evaluator = RetrievalEvaluator(searcher, mapping)

    recall = evaluator.recall_at_k(K_VALUES, sample_size=EVAL_SAMPLE_SIZE)
    print("Recall@K:", recall)

    precision = evaluator.precision_at_k(K_VALUES, sample_size=EVAL_SAMPLE_SIZE)
    print("Precision@K:", precision)

    print(f"\nRunning latency benchmark on {LATENCY_SAMPLE_SIZE} queries...")
    latency = evaluator.latency_benchmark(sample_size=LATENCY_SAMPLE_SIZE)
    print("Latency:", latency)

    print(f"\nRunning duplicate detection (threshold={DUPLICATE_THRESHOLD})...")
    embeddings = np.load(EMBEDDINGS_PATH)
    detector = DuplicateDetector(searcher.index, embeddings, mapping, threshold=DUPLICATE_THRESHOLD)
    duplicates = detector.find_duplicates()
    print(f"Found {len(duplicates)} near-duplicate pairs")
    for d in duplicates[:5]:
        print(f"  {d['similarity']:.4f}  {d['image_a']}  <->  {d['image_b']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "config": {
            "k_values": K_VALUES,
            "eval_sample_size": EVAL_SAMPLE_SIZE,
            "latency_sample_size": LATENCY_SAMPLE_SIZE,
            "duplicate_threshold": DUPLICATE_THRESHOLD,
        },
        "recall": recall,
        "precision": precision,
        "latency": latency,
        "duplicate_pairs_found": len(duplicates),
        "duplicate_pairs": duplicates[:50],
    }

    report_path = REPORT_DIR / f"eval_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved evaluation report to {report_path}")


if __name__ == "__main__":
    main()