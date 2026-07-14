import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class EmbeddingGenerator:
    """
    Generates CLIP image and text embeddings for a set of records, and saves
    them to disk as .npy files plus an index->record mapping.

    Uses a pretrained CLIP model as-is (no fine-tuning). CLIPProcessor
    handles resize/crop/normalization consistently with how the model was
    trained — no manual normalization happens here or upstream.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device}...")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # inference mode — disables dropout etc.
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    def _extract_features(self, output):
        """
        Helper method to handle both old and new transformers versions.
        - Old version: directly returns tensor
        - New version (5.x): returns BaseModelOutputWithPooling object
        """
        # Check if output is a tensor already (old version)
        if isinstance(output, torch.Tensor):
            return output
        
        # Check for common attribute names in new version
        if hasattr(output, 'pooler_output'):
            return output.pooler_output
        elif hasattr(output, 'image_embeds'):
            return output.image_embeds
        elif hasattr(output, 'text_embeds'):
            return output.text_embeds
        elif hasattr(output, 'last_hidden_state'):
            # Fallback: use last_hidden_state mean pooling
            return output.last_hidden_state.mean(dim=1)
        else:
            # If nothing works, try to convert to tensor
            return torch.tensor(output) if hasattr(output, 'shape') else output

    def generate_image_embeddings(self, records: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """
        Returns (embeddings array of shape [N, 512], list of records in the
        same order as the embeddings rows).
        """
        all_embeddings = []
        ordered_records = []

        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            images = []
            valid_batch = []

            for r in batch:
                try:
                    img = Image.open(r["processed_path"]).convert("RGB")
                    images.append(img)
                    valid_batch.append(r)
                except Exception as e:
                    print(f"  Skipping {r.get('processed_path')}: {e}")

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():  # no gradients needed — inference only
                output = self.model.get_image_features(**inputs)
                # Extract features from output (handles both old and new versions)
                features = self._extract_features(output)
                # L2 normalize so downstream cosine similarity == dot product
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())
            ordered_records.extend(valid_batch)

            done = min(i + self.batch_size, len(records))
            if done % (self.batch_size * 10) == 0 or done == len(records):
                print(f"  Image embeddings: {done}/{len(records)}")

        if not all_embeddings:
            return np.array([]).reshape(0, 512), []
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return embeddings, ordered_records

    def generate_text_embeddings(self, records: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """
        Returns (embeddings array of shape [N, 512], list of records in the
        same order as the embeddings rows).
        """
        all_embeddings = []
        ordered_records = []

        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            captions = [r["caption"] for r in batch]

            inputs = self.processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                output = self.model.get_text_features(**inputs)
                # Extract features from output (handles both old and new versions)
                features = self._extract_features(output)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())
            ordered_records.extend(batch)

            done = min(i + self.batch_size, len(records))
            if done % (self.batch_size * 10) == 0 or done == len(records):
                print(f"  Text embeddings: {done}/{len(records)}")

        if not all_embeddings:
            return np.array([]).reshape(0, 512), []
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return embeddings, ordered_records

    @staticmethod
    def save(embeddings: np.ndarray, records: list[dict], output_dir: str, prefix: str):
        """
        Saves embeddings as {prefix}_embeddings.npy and a matching
        {prefix}_mapping.json (row index -> record) so FAISS search results
        (which return integer indices) can be traced back to actual records.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / f"{prefix}_embeddings.npy", embeddings)

        mapping = {
            str(idx): {
                "image_path": r["image_path"],
                "caption": r["caption"],
                "processed_path": r.get("processed_path"),
            }
            for idx, r in enumerate(records)
        }
        with open(out / f"{prefix}_mapping.json", "w") as f:
            json.dump(mapping, f)

        print(f"Saved {embeddings.shape[0]} {prefix} embeddings to {out}/")