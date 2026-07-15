import json
import faiss
import numpy as np
import torch
from pathlib import Path

from app.services.embedding_generator import EmbeddingGenerator


class ImageSearcher:
    """
    Text-to-image search over a pre-built FAISS index.

    Reuses EmbeddingGenerator for CLIP model loading and text encoding, so
    the query is encoded identically to how the indexed embeddings were
    generated (same model, same normalization).
    """

    def __init__(
        self,
        index_path: str = "index/image_index.faiss",
        mapping_path: str = "embeddings/image_mapping.json",
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | None = None,
    ):
        self.index = faiss.read_index(index_path)

        with open(mapping_path) as f:
            self.mapping = json.load(f)

        # batch_size is irrelevant here — queries are encoded one at a time
        self.embedder = EmbeddingGenerator(model_name=model_name, device=device, batch_size=1)

    def _encode_query(self, query_text: str) -> np.ndarray:
        inputs = self.embedder.processor(
            text=[query_text], return_tensors="pt", padding=True, truncation=True
        ).to(self.embedder.device)

        with torch.no_grad():
            output = self.embedder.model.get_text_features(**inputs)
            features = self.embedder._extract_features(output)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype("float32")

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        query_emb = self._encode_query(query_text)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            record = self.mapping.get(str(idx), {})
            results.append({
                "image_path": record.get("image_path"),
                "caption": record.get("caption"),
                "score": float(score),
            })
        return results
