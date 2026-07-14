import faiss
import numpy as np


class IndexBuilder:
    """Builds and manages a FAISS index for CLIP image embeddings."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def build(self, embeddings: np.ndarray) -> faiss.Index:
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        return self.index

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @staticmethod
    def load(path: str) -> faiss.Index:
        return faiss.read_index(path)