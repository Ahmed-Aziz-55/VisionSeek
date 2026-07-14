# VisionSeek

VisionSeek is a CLIP + FAISS image search project. This repository currently
focuses on the data pipeline that turns a raw image-caption dataset into a
validated, inspectable, preprocessed set of records, generates CLIP
embeddings, and builds a searchable FAISS index.

## Current pipeline

The pipeline is split into small, single-purpose components:

1. `DatasetLoader` reads the raw pipe-separated file and returns dictionaries.
2. `DatasetValidator` applies pass/fail rules and separates valid from rejected
   records.
3. `DatasetInspector` computes descriptive statistics only.
4. `Preprocessor` resizes and normalizes images, saving processed files with
   UUID-based names.
5. `ImageRecord` defines the validated record schema with Pydantic.
6. CLIP embedding generation produces normalized image and text embeddings
   (512-dim, from `openai/clip-vit-base-patch32`).
7. `IndexBuilder` builds a FAISS `IndexFlatIP` index from the normalized
   image embeddings for cosine-similarity search.

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
DatasetValidator.validate()   → (valid_records, rejected_records)
      │
      ├──► DatasetInspector.summary()   → total rows, category counts, caption length
      │
      ▼
Preprocessor                  → resize, optional normalization, processed_path
      │
      ▼
ImageRecord                    → validated schema for a single record
      │
      ▼
CLIP embedding generation      → normalized image_embeddings.npy, text_embeddings.npy
      │
      ▼
IndexBuilder.build()            → FAISS IndexFlatIP, saved as index/image_index.faiss
```

## Components

- [app/dataset/loader.py](app/dataset/loader.py) contains `DatasetLoader`,
  which only reads and flags malformed rows.
- [app/dataset/validator.py](app/dataset/validator.py) contains
  `DatasetValidator`, which checks empty captions, empty categories, and image
  existence on disk.
- [app/dataset/inspector.py](app/dataset/inspector.py) contains
  `DatasetInspector`, which reports statistics without mutating records.
- [app/dataset/preprocessor.py](app/dataset/preprocessor.py) contains
  `Preprocessor`, which resizes images, optionally normalizes them for CLIP,
  and writes processed JPEGs to disk.
- [app/models/image_record.py](app/models/image_record.py) contains
  `ImageRecord`, the Pydantic model for a valid record.
- [app/scripts/generate_embeddings.py](app/scripts/generate_embeddings.py)
  loads the CLIP model and generates normalized image and text embeddings
  from the preprocessed manifest.
- [app/index/index_builder.py](app/index/index_builder.py) contains
  `IndexBuilder`, which builds, saves, and loads a FAISS index from image
  embeddings.
- [app/scripts/build_index.py](app/scripts/build_index.py) loads the saved
  image embeddings and builds the FAISS index via `IndexBuilder`.

## Validation rules

The current validator rejects records when:

- the loader already tagged the row with `_error`
- `caption` is empty or whitespace only
- `category` is empty or whitespace only
- the image file does not exist at `base_image_dir / image_path`

Each rejected record carries a `_reasons` list so failures remain visible during
analysis.

## Preprocessing behavior

The preprocessor currently:

- resizes images to a target size of 224 x 224 by default
- can preserve aspect ratio with padding
- can apply CLIP-style normalization
- saves outputs as JPEG files with UUID-based filenames
- stores the processed output path in `processed_path`
- raises an error if the failure rate exceeds the configured threshold

## Embedding generation

`generate_embeddings.py`:

- loads CLIP (`openai/clip-vit-base-patch32`) on CPU
- generates image embeddings and text embeddings in batches
- L2-normalizes every embedding (unit vectors, norm ≈ 1.0) so that FAISS
  inner-product search is equivalent to cosine similarity
- saves both as `.npy` files under `embeddings/`, shape `(N, 512)`

## Index building

`IndexBuilder`:

- wraps a FAISS `IndexFlatIP` index (inner product, exact search)
- expects embeddings to already be L2-normalized — it does not normalize
  internally
- casts embeddings to `float32` before adding, since FAISS requires it
- can save/load the index to/from disk via `faiss.write_index` /
  `faiss.read_index`

## Dataset format

Input is a plain text file with one record per line, using pipe separators:

```text
image_path|caption|category
datasets/Images/example.jpg|a short caption for the image|image
```

## Example usage

```python
from app.dataset.loader import DatasetLoader
from app.dataset.validator import DatasetValidator
from app.dataset.inspector import DatasetInspector
from app.dataset.preprocessor import Preprocessor
from app.index.index_builder import IndexBuilder
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

# After running generate_embeddings.py:
img_emb = np.load("embeddings/image_embeddings.npy")
builder = IndexBuilder(dimension=img_emb.shape[1])
index = builder.build(img_emb)
builder.save("index/image_index.faiss")
```

## Project layout

```text
app/
  dataset/        loader, validator, inspector, preprocessor
  models/         Pydantic record schema
  index/          FAISS IndexBuilder
  scripts/        utility scripts: generate_embeddings, build_index, dataset checks
  docs/           architecture notes and design decisions
data/raw/         source dataset text files
datasets/Images/   source image data and CSV inputs
processed/images/ processed image outputs
embeddings/       saved image and text embeddings (.npy)
index/            saved FAISS index files
```

## Related notes

- [app/docs/Decisions.md](app/docs/Decisions.md) records the architecture
  decisions behind the current pipeline design.