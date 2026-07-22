# VisionSeek

VisionSeek is a CLIP + FAISS text-to-image search project. It turns a raw
image-caption dataset into a validated, preprocessed set of records,
generates CLIP embeddings, builds a searchable FAISS index, and supports
text-to-image search with a benchmarked evaluation suite.

## Current pipeline

The pipeline is split into small, single-purpose components:

1. `DatasetLoader` reads the raw pipe-separated file and returns dictionaries.
2. `DatasetValidator` applies pass/fail rules (including Pydantic schema
   validation via `ImageRecord`) and separates valid from rejected records.
3. `DatasetInspector` computes descriptive statistics only.
4. `Preprocessor` resizes images, saving processed files with UUID-based
   names.
5. `ImageRecord` defines the validated record schema with Pydantic.
6. `EmbeddingGenerator` produces normalized image and text CLIP embeddings
   (768-dim, from `openai/clip-vit-large-patch14`).
7. `IndexBuilder` builds a FAISS `IndexFlatIP` index from the normalized
   image embeddings for cosine-similarity search.
8. `ImageSearcher` runs text-to-image search against the built index.
9. `RetrievalEvaluator` and `DuplicateDetector` benchmark the index:
   Recall@K, Precision@K, search latency, and near-duplicate detection.

Malformed rows are not silently dropped. The loader tags them with `_error`, the
validator collects rejection reasons in `_reasons`, and the inspector excludes
loader errors from summary statistics.

## Pipeline flow

```text
raw dataset file
      │
      ▼
DatasetLoader.load()          → list[dict], malformed lines flagged with _error
      │
      ▼
DatasetValidator.validate()   → (valid_records, rejected_records), Pydantic-checked
      │
      ├──► DatasetInspector.summary()   → total rows, category counts, caption length
      │
      ▼
Preprocessor                  → resize, processed_path, UUID filenames
      │
      ▼
EmbeddingGenerator             → normalized image_embeddings.npy, text_embeddings.npy (768-dim)
      │
      ▼
IndexBuilder.build()            → FAISS IndexFlatIP, saved as index/image_index.faiss
      │
      ▼
ImageSearcher.search()          → text query → top-K matching images
      │
      ▼
RetrievalEvaluator / DuplicateDetector → Recall@K, Precision@K, latency, duplicates
```

## Components

- [app/dataset/loader.py](app/dataset/loader.py) — `DatasetLoader`, reads
  and flags malformed rows.
- [app/dataset/validator.py](app/dataset/validator.py) — `DatasetValidator`,
  checks empty captions, empty categories, image existence, and Pydantic
  schema validity via `ImageRecord`.
- [app/dataset/inspector.py](app/dataset/inspector.py) — `DatasetInspector`,
  reports statistics without mutating records.
- [app/dataset/preprocessor.py](app/dataset/preprocessor.py) — `Preprocessor`,
  resizes images and writes processed JPEGs to disk.
- [app/models/image_record.py](app/models/image_record.py) — `ImageRecord`,
  the Pydantic model for a valid record.
- [app/services/embedding_generator.py](app/services/embedding_generator.py) —
  `EmbeddingGenerator`, loads CLIP and generates normalized image/text embeddings.
- [app/scripts/generate_embeddings.py](app/scripts/generate_embeddings.py) —
  runs `EmbeddingGenerator` over the preprocessed manifest.
- [app/index/index_builder.py](app/index/index_builder.py) — `IndexBuilder`,
  builds, saves, and loads a FAISS index from image embeddings.
- [app/scripts/build_index.py](app/scripts/build_index.py) — builds the
  FAISS index via `IndexBuilder`.
- [app/services/searcher.py](app/services/searcher.py) — `ImageSearcher`,
  encodes a text query and searches the FAISS index.
- [app/scripts/search_demo.py](app/scripts/search_demo.py) — interactive
  CLI for trying searches.
- [app/evaluation/metrics.py](app/evaluation/metrics.py) — `RetrievalEvaluator`,
  computes Recall@K, Precision@K, and search latency.
- [app/evaluation/duplicate_detector.py](app/evaluation/duplicate_detector.py) —
  `DuplicateDetector`, finds near-duplicate images via the FAISS index.
- [app/scripts/run_evaluation.py](app/scripts/run_evaluation.py) — runs the
  full evaluation suite and saves a JSON report.
- [app/core/logging_config.py](app/core/logging_config.py) — centralized
  logging setup used by all pipeline scripts.

## Validation rules

The current validator rejects records when:

- the loader already tagged the row with `_error`
- `caption` is empty or whitespace only
- `category` is empty or whitespace only
- the image file does not exist at `base_image_dir / image_path`
- the record fails `ImageRecord` Pydantic schema validation

Each rejected record carries a `_reasons` list so failures remain visible during
analysis.

## Preprocessing behavior

The preprocessor currently:

- resizes images to a target size of 224 x 224 by default
- can preserve aspect ratio with padding
- saves outputs as JPEG files with UUID-based filenames
- stores the processed output path in `processed_path`
- raises an error if the failure rate exceeds the configured threshold

## Embedding generation

`EmbeddingGenerator` (via `generate_embeddings.py`):

- loads CLIP (`openai/clip-vit-large-patch14`) on CPU
- generates image embeddings and text embeddings in batches
- L2-normalizes every embedding (unit vectors, norm ≈ 1.0) so that FAISS
  inner-product search is equivalent to cosine similarity
- saves both as `.npy` files under `embeddings/`, shape `(N, 768)`

Originally used `openai/clip-vit-base-patch32` (512-dim); upgraded to
ViT-Large after observing weak text-image alignment on multi-concept
queries — see [Decision 13](app/docs/Decisions.md) for the full analysis.

## Index building

`IndexBuilder`:

- wraps a FAISS `IndexFlatIP` index (inner product, exact search)
- expects embeddings to already be L2-normalized — it does not normalize
  internally
- casts embeddings to `float32` before adding, since FAISS requires it
- can save/load the index to/from disk via `faiss.write_index` /
  `faiss.read_index`

## Search

`ImageSearcher`:

- loads the saved FAISS index and image mapping
- reuses `EmbeddingGenerator`'s CLIP model to encode the text query
  identically to how the indexed embeddings were generated
- returns top-K matches as `{image_path, caption, score}`

Try it interactively:

```bash
python -m app.scripts.search_demo
```

## Results

Evaluated on the full dataset (31,783 caption→image queries), using each
caption as a query and checking whether its original image is retrieved.

| Metric | Value |
|---|---|
| Recall@1 | 0.498 |
| Recall@5 | 0.733 |
| Recall@10 | 0.809 |
| Mean search latency | 102.3 ms |
| p95 latency | 150.3 ms |
| Near-duplicate images found (similarity ≥ 0.98) | 4 pairs |

Full methodology and analysis in
[app/docs/Decisions.md](app/docs/Decisions.md).

Reproduce with:

```bash
python -m app.scripts.run_evaluation
```

## Dataset format

Input is a plain text file with one record per line, using pipe separators:

```text
image_path|caption|category
datasets/Images/example.jpg|a short caption for the image|image
```

## Setup

Install runtime dependencies (PyTorch CPU build requires an extra index):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

For development (linting, formatting, tests):

```bash
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Example usage

```python
from app.dataset.loader import DatasetLoader
from app.dataset.validator import DatasetValidator
from app.dataset.inspector import DatasetInspector
from app.dataset.preprocessor import Preprocessor
from app.services.embedding_generator import EmbeddingGenerator
from app.index.index_builder import IndexBuilder
from app.services.searcher import ImageSearcher
import numpy as np

records = DatasetLoader("data/raw/dataset.txt").load()
valid_records, rejected_records = DatasetValidator(
    records,
    base_image_dir=".",
).validate()

stats = DatasetInspector(valid_records).summary()
print(stats)

preprocessor = Preprocessor(output_dir="processed/images")
processed_records, failed_records = preprocessor.process(valid_records, base_image_dir=".")

# After running generate_embeddings.py and build_index.py:
searcher = ImageSearcher()
results = searcher.search("a dog running on the beach", top_k=5)
```

## Project layout

```text
app/
  dataset/        loader, validator, inspector, preprocessor
  models/         Pydantic record schema
  services/       EmbeddingGenerator, ImageSearcher
  index/          FAISS IndexBuilder
  evaluation/      RetrievalEvaluator, DuplicateDetector, saved reports/
  core/           logging_config
  scripts/        generate_embeddings, build_index, run_evaluation, search_demo, reprocess_images
  docs/           architecture notes and design decisions
data/raw/         source dataset text files
datasets/Images/   source image data and CSV inputs
processed/images/ processed image outputs
embeddings/       saved image and text embeddings (.npy)
index/            saved FAISS index files
logs/             application logs (gitignored)
```

## Related notes

- [app/docs/Decisions.md](app/docs/Decisions.md) records the architecture
  decisions behind the current pipeline design.





  ## Docker

Build the image:

```bash
docker build -t visionseek .
```

Run the interactive search demo (mounts the dataset, embeddings, index, and
Hugging Face model cache from the host so the image stays small and the
CLIP model isn't re-downloaded on every run):

```bash
docker run -it \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/index:/app/index \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  visionseek
```

Run any other script instead of the default search demo, e.g. the
evaluation suite:

```bash
docker run -it \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/index:/app/index \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  visionseek python -m app.scripts.run_evaluation
```

The container runs as a non-root user (`appuser`) and does not bundle the
image dataset, embeddings, or FAISS index — those are mounted at runtime
via `-v`, keeping the image itself small (code + dependencies only).

