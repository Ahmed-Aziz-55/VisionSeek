"""
app/scripts/run_evaluation.py

Runs the full VisionSeek evaluation suite:
  - Recall@K / Precision@K (text-to-image retrieval quality)
  - Duplicate detection (near-duplicate images in the dataset)
  - Search latency benchmark

Saves a single JSON report under app/evaluation/reports/ for the
internship evaluation write-up.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from app.core.logging_config import setup_logging
from app.services.searcher import ImageSearcher
from app.evaluation.metrics import RetrievalEvaluator
from app.evaluation.duplicate_detector import DuplicateDetector

setup_logging()
logger = logging.getLogger(__name__)

# ---- Config ----
K_VALUES = [1, 5, 10]
EVAL_SAMPLE_SIZE = None      # None = full dataset
LATENCY_SAMPLE_SIZE = 100
DUPLICATE_THRESHOLD = 0.98

MAPPING_PATH = "embeddings/image_mapping.json"
EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"
REPORT_DIR = Path("app/evaluation/reports")


def main():
    logger.info("Loading searcher (FAISS index + CLIP model)...")
    searcher = ImageSearcher()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    logger.info(f"Loaded mapping with {len(mapping)} records")

    logger.info(f"Running retrieval evaluation on {EVAL_SAMPLE_SIZE or 'ALL'} queries...")
    evaluator = RetrievalEvaluator(searcher, mapping)

    recall = evaluator.recall_at_k(K_VALUES, sample_size=EVAL_SAMPLE_SIZE)
    logger.info(f"Recall@K: {recall}")

    precision = evaluator.precision_at_k(K_VALUES, sample_size=EVAL_SAMPLE_SIZE)
    logger.info(f"Precision@K: {precision}")

    logger.info(f"Running latency benchmark on {LATENCY_SAMPLE_SIZE} queries...")
    latency = evaluator.latency_benchmark(sample_size=LATENCY_SAMPLE_SIZE)
    logger.info(f"Latency: {latency}")

    logger.info(f"Running duplicate detection (threshold={DUPLICATE_THRESHOLD})...")
    embeddings = np.load(EMBEDDINGS_PATH)
    detector = DuplicateDetector(searcher.index, embeddings, mapping, threshold=DUPLICATE_THRESHOLD)
    duplicates = detector.find_duplicates()
    logger.info(f"Found {len(duplicates)} near-duplicate pairs")
    for d in duplicates[:5]:
        logger.info(f"  {d['similarity']:.4f}  {d['image_a']}  <->  {d['image_b']}")

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

    logger.info(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()