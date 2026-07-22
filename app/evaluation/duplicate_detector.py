import numpy as np


class DuplicateDetector:
    """
    For each image, queries the FAISS index for its 2 nearest neighbors
    (itself + closest other). If the second-best match's similarity exceeds
    `threshold`, it's flagged as a near-duplicate pair.

    Uses the FAISS index directly instead of a raw N x N similarity matrix —
    a full pairwise matrix for 31,783 images would need ~4GB and O(N^2)
    comparisons; batched FAISS search does the same job in seconds without
    holding that matrix in memory.
    """

    def __init__(self, index, embeddings: np.ndarray, mapping: dict, threshold: float = 0.98):
        self.index = index
        self.embeddings = embeddings.astype(np.float32)
        self.mapping = mapping
        self.threshold = threshold

    def find_duplicates(self, batch_size: int = 512) -> list[dict]:
        n = self.embeddings.shape[0]
        seen_pairs = set()
        duplicates = []

        for start in range(0, n, batch_size):
            batch = self.embeddings[start:start + batch_size]
            scores, indices = self.index.search(batch, 2)

            for row_offset, (row_scores, row_indices) in enumerate(zip(scores, indices)):
                query_idx = start + row_offset
                for score, match_idx in zip(row_scores, row_indices):
                    if match_idx == query_idx or match_idx == -1:
                        continue
                    if score < self.threshold:
                        continue

                    pair = tuple(sorted((query_idx, int(match_idx))))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    duplicates.append({
                        "idx_a": pair[0],
                        "idx_b": pair[1],
                        "similarity": float(score),
                        "image_a": self.mapping[str(pair[0])]["image_path"],
                        "image_b": self.mapping[str(pair[1])]["image_path"],
                    })

        return sorted(duplicates, key=lambda d: -d["similarity"])