
import time
import numpy as np
from app.services.searcher import ImageSearcher


class RetrievalEvaluator:
    """
    Evaluates text-to-image retrieval quality of a built FAISS index using
    the caption -> original image pairing as ground truth.
    """

    def __init__(self, searcher: ImageSearcher, mapping: dict):
        self.searcher = searcher
        self.mapping = mapping  # idx(str) -> {"image_path", "caption", "processed_path"}

    def recall_at_k(self, k_values: list[int], sample_size: int | None = None) -> dict:
        """
        For each ground-truth (caption, image) pair: encode the caption as a
        query, check if the paired image lands in the top-K results.

        Recall@K = (# queries where correct image found in top-K) / (# queries)
        """
        items = list(self.mapping.items())
        if sample_size:
            items = items[:sample_size]

        max_k = max(k_values)
        hits = {k: 0 for k in k_values}

        for idx_str, record in items:
            results = self.searcher.search(record["caption"], top_k=max_k)
            retrieved_paths = [r["image_path"] for r in results]

            for k in k_values:
                if record["image_path"] in retrieved_paths[:k]:
                    hits[k] += 1

        total = len(items)
        return {f"recall@{k}": hits[k] / total for k in k_values}

    def precision_at_k(self, k_values: list[int], sample_size: int | None = None) -> dict:
        """
        Precision@K for a single-relevant-item task: 1/K if the correct
        image is found within top-K, else 0 — averaged across queries.
        (With one relevant doc per query this equals Recall@K / K, but
        computing it explicitly keeps the metric self-documenting.)
        """
        items = list(self.mapping.items())
        if sample_size:
            items = items[:sample_size]

        max_k = max(k_values)
        scores = {k: [] for k in k_values}

        for idx_str, record in items:
            results = self.searcher.search(record["caption"], top_k=max_k)
            retrieved_paths = [r["image_path"] for r in results]

            for k in k_values:
                found = 1 if record["image_path"] in retrieved_paths[:k] else 0
                scores[k].append(found / k)

        return {f"precision@{k}": float(np.mean(scores[k])) for k in k_values}

    def latency_benchmark(self, sample_size: int = 100, top_k: int = 5) -> dict:
        """Measures search() wall-clock latency (ms) over sample_size queries."""
        items = list(self.mapping.items())[:sample_size]
        latencies = []

        for idx_str, record in items:
            start = time.perf_counter()
            self.searcher.search(record["caption"], top_k=top_k)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        return {
            "mean_ms": round(float(latencies.mean()), 2),
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "sample_size": sample_size,
        }