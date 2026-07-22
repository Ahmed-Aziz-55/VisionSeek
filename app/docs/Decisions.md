# VisionSeek — Data Pipeline: Architecture Decisions

This document records the key design decisions made while building VisionSeek's
data pipeline, along with the reasoning behind each choice. Intended for internship
evaluation and as a personal reference.

---

## 1. Pipeline structure follows Single Responsibility Principle

The pipeline is split into distinct, single-purpose components rather than one
monolithic script:

| Component | Responsibility |
|---|---|
| `DatasetLoader` | Reads raw dataset file, returns raw dict rows. Does **not** validate. |
| `DatasetValidator` | Applies pass/fail rules, splits records into valid vs rejected. |
| `DatasetInspector` | Computes descriptive statistics only. Does **not** filter or reject. |
| `ImageRecord` | Pydantic schema — defines and enforces the shape of a valid record. |
| `Preprocessor` | Resizes/normalizes images, assigns `processed_path`. |
| `IndexBuilder` | Builds/saves/loads a FAISS index from embeddings. Does **not** generate embeddings. |

**Reasoning:** each stage has exactly one reason to change. If the dataset file
format changes, only `DatasetLoader` is touched. If validation rules change
(e.g. a new required field), only `DatasetValidator` changes. This keeps each
class small, testable in isolation, and easy to reason about — important for
being able to explain the architecture without depending on an AI assistant
during evaluation.

---

## 2. Loader does not validate — it flags and defers

`DatasetLoader.load()` reads a pipe-separated (`|`) file line by line. If a line
doesn't split into exactly 3 fields, the loader does **not** raise or drop the
row. Instead it appends a dict with an `_error` key describing the problem:

```python
records.append({"_error": f"line {line_num}: expected 3 fields, got {len(parts)}"})
```

**Reasoning:** the Loader's job is *reading*, not *deciding what counts as
acceptable data*. That decision belongs to the Validator. Flagging instead of
silently dropping means no row disappears without a trace — every malformed
line is visible downstream and shows up in the rejected set with a reason.

---

## 3. Validator centralizes all pass/fail rules

`DatasetValidator.validate()` is the single place where a record is judged:

- Rows already flagged with `_error` by the Loader are rejected immediately.
- Empty `caption` or empty `category` (after stripping whitespace) → rejected.
- Image file must exist on disk at `base_image_dir / image_path` → rejected
  otherwise.
- Every rejection carries a `_reasons` list, so a row can fail for multiple
  reasons at once and all of them are visible.

**Reasoning:** having exactly one place where "is this record usable?" is
decided avoids validation logic leaking into other components. It also makes
the rejected set self-documenting — every rejected row explains *why* it was
rejected, which matters for error analysis during evaluation.

---

## 4. Inspector is read-only by design

`DatasetInspector` deliberately has no method that filters, mutates, or
rejects records — it only computes stats (`total_rows`, category distribution,
average caption length). It also excludes `_error` rows automatically before
computing stats, so malformed rows don't skew descriptive numbers.

**Reasoning:** keeping inspection separate from validation means dataset stats
can be pulled at any pipeline stage (before or after validation) without side
effects. It's a pure reporting tool, not a gate.

---

## 5a. `ImageRecord` lives in `app/models/`, not `app/dataset/`

The pipeline processing components (`loader.py`, `validator.py`,
`inspector.py`, `preprocessor.py`) live in `app/dataset/`, but the
`ImageRecord` schema itself lives in `app/models/`. Processing modules import
it from there:

```python
from app.models.image_record import ImageRecord
```

**Reasoning:** `models/` holds the definition of what valid data looks like
(the schema), while `dataset/` holds the logic that reads, checks, and
transforms data. Keeping the schema in one place — rather than duplicating it
per module — means there is a single source of truth for the record shape.
Any change to required fields only has to happen once.

---

## 5. Pydantic over dataclass for `ImageRecord`

`ImageRecord` is defined as a Pydantic `BaseModel` rather than a `@dataclass`.

**Reasoning:** Pydantic performs runtime type validation — a dataclass only
checks types statically (or not at all at runtime). Since dataset rows come
from an external file, runtime validation is necessary: a malformed row could
otherwise pass silently into the pipeline as a "valid" Python object with the
wrong types. The custom `field_validator` on `caption`/`category` additionally
rejects whitespace-only strings, which plain type-checking wouldn't catch.

---

## 6. `processed_path` is optional at the schema level

`ImageRecord.processed_path` is typed as `str | None = None`.

**Reasoning:** a record is valid as raw metadata before preprocessing has run.
Making `processed_path` optional lets the same schema represent a record at
two different pipeline stages (pre- and post-preprocessing) without needing a
second model.

---￼
ahmed

## 7. UUID-based filenames in Preprocessor

`Preprocessor.process()` saves every processed image under a
`uuid.uuid4().hex` filename instead of keeping the original filename.

**Reasoning:** original filenames can collide across categories (e.g.
`img_001.jpg` existing in both a "cat" folder and a "car" folder). Writing
both into the same flat `output_dir` under their original names risks
silently overwriting one with the other. UUIDs are effectively guaranteed
unique, so every processed file gets its own safe slot regardless of what
the source was named.

---

## 8. Configurable failure threshold in Preprocessor

`Preprocessor` accepts a `failure_threshold` (default `0.1` = 10%). After
processing all records, if the fraction that failed (corrupt file, missing
file, unreadable image, etc.) exceeds this threshold, the pipeline raises
`RuntimeError` and stops rather than silently continuing.

**Reasoning:** a handful of bad images failing is normal and shouldn't halt
the pipeline. But a *high* failure rate usually signals something systemic —
a bad download batch, a wrong `base_image_dir`, a broken upstream export —
and continuing on mostly-corrupt data would poison downstream embeddings and
training silently. The threshold turns a silent data-quality problem into a
loud, immediate one that has to be investigated before proceeding.

---

## 9. Embeddings are L2-normalized at generation time, not at index time

`generate_embeddings.py` normalizes every image and text embedding
(dividing by its L2 norm, so `‖v‖ ≈ 1.0`) immediately after extracting it
from CLIP, before saving to `.npy`.

**Reasoning:** normalization is a property of the embedding itself, not of
how it's later indexed or searched. Doing it once at generation time — rather
than repeating it in the index-building step, the search step, or anywhere
else that touches the embeddings — means every downstream consumer can rely
on embeddings already being unit vectors, with no risk of forgetting to
normalize in one of several places. Verified empirically: `np.linalg.norm`
on saved embeddings returns ≈1.0 across sampled rows for both image and text
embeddings.

---

## 10. `IndexBuilder` uses `IndexFlatIP`, not `IndexFlatL2`

`IndexBuilder` wraps FAISS's `IndexFlatIP` (inner product, exact search)
rather than `IndexFlatL2` (Euclidean distance).

**Reasoning:** since embeddings are already L2-normalized (see Decision 9),
inner product between two vectors is mathematically equivalent to their
cosine similarity. `IndexFlatIP` gives that similarity directly, without
needing a separate normalization or conversion step at search time.
`IndexFlatL2` would still produce a *usable* ranking on normalized vectors
(Euclidean distance and cosine similarity are monotonically related for unit
vectors), but `IndexFlatIP` is the more direct, standard choice for
CLIP-style embeddings and avoids relying on that indirect relationship.

`IndexBuilder` itself does **not** normalize embeddings — it assumes they
arrive already normalized, keeping normalization a single-responsibility
concern of the embedding-generation step, not the indexing step.

---

## 11. `IndexBuilder` separates build/save/load from embedding generation

`IndexBuilder` lives in `app/index/`, separate from `app/scripts/generate_embeddings.py`.
It only knows how to build a FAISS index from an existing embeddings array,
save it to disk, and load it back — it has no knowledge of CLIP or how
embeddings were produced.

**Reasoning:** this follows the same SRP boundary as the rest of the
pipeline (Decision 1). Embedding generation and index building are separate
concerns with separate reasons to change — e.g. switching CLIP model
variants only touches `generate_embeddings.py`; switching FAISS index type
(e.g. to an approximate index for larger scale) only touches `IndexBuilder`.

---

## 12. FAISS index built and sanity-verified

The image index was built via `app/scripts/build_index.py`, producing
`index/image_index.faiss` with all 31,783 image embeddings. Verified with
a self-query sanity check: searching the index using image 0 as the query
returns image 0 itself as the top match with similarity ~1.0, and the
remaining top-5 matches show a reasonable descending similarity ranking —
confirming the index is not returning arbitrary or randomly-ordered results.


---

## 13. Upgraded from CLIP ViT-B/32 to ViT-L/14 after observing weak text-image alignment

The initial embedding pipeline used `openai/clip-vit-base-patch32` (512-dim).
A qualitative test — searching "white cat" — returned zero genuine white-cat
results in the top-5, despite the dataset containing 29 images whose captions
mention both "white" and "cat" (verified via word-boundary regex match on
`image_mapping.json`, after ruling out substring false positives like
"catch" and "CAT construction equipment"). Since 29 relevant images existed
but none surfaced, the issue was diagnosed as a model limitation rather than
a dataset limitation — base-size CLIP is known to weight individual
attributes (e.g. color) over object identity in multi-concept queries.

Switched to `openai/clip-vit-large-patch14` (768-dim). Re-generated all
image and text embeddings and rebuilt the FAISS index. The same "white cat"
query then returned 3 genuine white-cat matches in the top-5, up from 0/5 —
confirming the upgrade measurably improved text-image semantic alignment.

**Reasoning:** diagnosing "dataset vs. model" before changing anything
avoided wasting effort tuning the wrong component. Only after confirming
sufficient ground-truth examples existed in the dataset was the model
treated as the bottleneck. `EmbeddingGenerator` and `IndexBuilder` required
no code changes for this swap — `model_name` is a constructor parameter and
embedding dimension is read dynamically (`img_emb.shape[1]`), which is a
direct payoff of the SRP-based design from Decisions 1 and 11.

**Trade-off:** ViT-L/14 is significantly slower on CPU (full-dataset
embedding generation took well over an hour for 31,783 images, vs. a few
minutes for ViT-B/32) and requires a larger download (~1.7GB vs ~600MB).
Acceptable here since embedding generation is a one-time offline step, not
part of the online search request path.




---

## 14. Evaluation suite built on self-retrieval ground truth (caption → original image)

`app/evaluation/metrics.py` (`RetrievalEvaluator`) and
`app/evaluation/duplicate_detector.py` (`DuplicateDetector`) were added to
produce the benchmark numbers required by the internship evaluation:
Recall@K, Precision@K, duplicate detection, and search latency.

**Ground truth choice:** the dataset has no separate relevance-labeled
query set. Instead, each (caption, image) pair already in the dataset is
treated as ground truth — the caption is used as the search query, and the
image it was originally written for is the single relevant result. This
"self-retrieval" evaluation is the standard approach for CLIP-style
text-to-image retrieval benchmarks (same principle as Flickr30k/COCO
retrieval evaluation), and required no new labeled data.

**Duplicate detection implementation:** initially considered a brute-force
N×N cosine similarity matrix (`embeddings @ embeddings.T`), but for 31,783
images this would require a ~4GB float32 matrix and ~1 billion pairwise
comparisons. Instead, `DuplicateDetector` reuses the already-built FAISS
`IndexFlatIP` index — for each image it queries the index for its top-2
nearest neighbors (itself + closest other) in batches, which finds all
near-duplicate pairs without ever materializing a full pairwise matrix.

### Results (50-sample quick run, `clip-vit-large-patch14`)

| Metric | Value |
|---|---|
| Recall@1 | 0.58 |
| Recall@5 | 0.84 |
| Recall@10 | 0.92 |
| Precision@1 | 0.58 |
| Precision@5 | 0.168 |
| Precision@10 | 0.092 |
| Mean search latency | 124.84 ms |
| p95 latency | 173.1 ms |
| Near-duplicate pairs found (threshold=0.98) | 4 |

**Precision@K note:** with exactly one relevant image per query,
Precision@K is mathematically tied to Recall@K / K (e.g. 0.84/5 ≈ 0.168).
This isn't a flaw in the metric — it's an expected property of a
single-relevant-item retrieval task, and is documented explicitly so it can
be explained during evaluation rather than mistaken for a bug.

**Duplicate finding:** one pair (`2851198725.jpg`, `3050606344.jpg`) scored
similarity 1.0000 — a near-exact visual duplicate under two different
filenames in the source dataset. This is a genuine data-quality finding
about the dataset, not a pipeline defect.

**Trade-off:** the 50-sample run is a fast sanity check, not the final
reported number — each query requires a live CLIP text encode + FAISS
search, so evaluating the full ~31K queries would take significantly
longer. A larger sample (e.g. 500) will be run for the final reported
metrics before submission.



****************************************************************
### Final results (full dataset, 31,783 queries)

| Metric | Value |
|---|---|
| Recall@1 | 0.4982 |
| Recall@5 | 0.7326 |
| Recall@10 | 0.8086 |
| Precision@1 | 0.4982 |
| Precision@5 | 0.1465 |
| Precision@10 | 0.0809 |
| Mean search latency | 102.33 ms |
| p95 latency | 150.3 ms |
| p99 latency | 166.0 ms |
| Near-duplicate pairs found (threshold=0.98) | 4 (identical to the 50-sample run) |

**50-sample vs full-dataset comparison:** the initial 50-sample sanity
check reported Recall@1=0.58, noticeably higher than the full-dataset
Recall@1=0.498. This is a concrete example of small-sample optimistic
bias — a random 50-query subset happened to contain a disproportionate
share of easy queries. This is precisely why the final reported metric is
the full 31,783-query run rather than the quick sample: sampling error on
a set this small was large enough to meaningfully overstate performance.
The duplicate-detection results were identical across both runs (same 4
pairs, same similarities), which is expected since duplicate detection is
deterministic and doesn't depend on query sampling.

*****************************************************************************


---

## 15. Removed unused `settings.py`, wired up `ImageRecord` in Validator, deleted stale `check_dataset.py`

A codebase audit (using `grep` to confirm actual usage rather than assuming)
found three issues:

1. **`app/configs/settings.py`** — defined a Pydantic `Settings` class but
   was imported nowhere in the codebase. Removed. A proper config module
   will be added when the FastAPI backend is built, scoped to what that
   backend actually needs.

2. **`ImageRecord` was imported but never instantiated.** `validator.py`
   and `inspector.py` both imported it, but no code ever called
   `ImageRecord(...)` — meaning the Pydantic runtime validation described
   in Decision 5 never actually ran during validation. Fixed by
   instantiating `ImageRecord` inside `DatasetValidator.validate()`,
   wrapped in a `try/except ValidationError`, after the existing manual
   checks pass. This makes the schema validation real rather than
   documented-but-unused. Removed the now-unnecessary import from
   `inspector.py`, which never needed the schema — it only reads plain
   dict fields for statistics.

3. **`app/scripts/check_dataset.py` was deleted.** It loaded the raw
   `datasets/Images/results.csv` directly through `DatasetLoader`, which
   expects the pipeline's `image_path|caption|category` format. The raw
   CSV's actual format is `image_name| comment_number| comment` — three
   pipe-separated fields, but with different semantics. Running it through
   `DatasetLoader` silently mismapped `comment_number` into the `caption`
   field and the actual caption text into `category`. This script predates
   `convert_dataset.py` (which correctly transforms the raw CSV into the
   pipeline's expected format) and was never updated after the pipeline's
   input format was finalized — a stale exploration script now fully
   superseded by `reprocess_images.py`.

**Reasoning:** all three were found by checking actual usage (`grep`)
rather than assuming a file's presence meant it was needed — the same
principle Decision 3 applies to validation itself: don't take correctness
on faith, verify it.
---

## 16. FastAPI backend wraps ImageSearcher, loaded once at startup via lifespan

`app/main.py` exposes `ImageSearcher` over HTTP with two endpoints:
`POST /search` and `GET /health`. The CLIP model and FAISS index are loaded
exactly once, at app startup, via FastAPI's `lifespan` context manager, and
stored on `app.state.searcher`.

**Reasoning:** loading CLIP takes measurable time (~1s+, longer without a
warm cache) and holds the model in memory. Loading it per-request would add
that cost to every single search call and needlessly reload the same
768-dim weights repeatedly. `lifespan` guarantees the load happens exactly
once, before the server accepts traffic, and `/health` reports whether the
searcher is ready (`searcher_ready: true/false`) so a caller — or a
container orchestrator's healthcheck — can distinguish "still starting up"
from "ready."

Request/response schemas (`app/schemas/search.py`) are kept separate from
`ImageSearcher`'s internal dict-based results, so the public API contract
(`SearchRequest`, `SearchResponse`, `SearchResult`) is explicit and doesn't
silently change if the internal search implementation changes its return
shape.

---

## 17. docker-compose.yml added for the API service; healthcheck avoids `curl`

`docker-compose.yml` runs the FastAPI service with `docker compose up`,
mounting the same volumes (`datasets/`, `embeddings/`, `index/`, the
Hugging Face cache) used by the plain `docker run` workflow, plus `logs/`.

The healthcheck initially used `curl -f http://localhost:8000/health`, but
the container's base image (`python:3.12-slim`) doesn't include `curl` —
only `libjpeg62-turbo` and `zlib1g` are installed (Decision on Dockerfile
system deps). Rather than add `curl` as an extra system dependency purely
for the healthcheck, the check uses Python's built-in `urllib` instead
(`python -c "import urllib.request; urllib.request.urlopen(...)"`), since
Python is already present in every layer of this image.

`version: '3.8'` was also removed from the compose file — it's a legacy
field ignored by Compose v2 (2.40.3, installed here) and produces a
deprecation warning with no functional purpose.

---

## 18. `pip install` given explicit retries and a longer timeout in Dockerfile

`RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt`
replaces the original bare `pip install` call.

**Reasoning:** a build failed with `ERROR: Could not find a version that
satisfies the requirement pydantic==2.13.4 (from versions: none)` —
`pydantic==2.13.4` is a real, correct pin (confirmed present in the local
venv), so this was not a real dependency error. Diagnosis (`ping`, `curl`
timing tests against PyPI and Debian mirrors) showed the network was slow
and lossy (~15 KB/s throughput, ~25% packet loss to a Debian mirror) —
pip's default 15-second timeout was too short for a slow connection to
even receive a response from PyPI before giving up, which pip then
misreported as "no matching version." Raising the timeout to 120s and
allowing 10 retries makes the build resilient to transient network
slowness without masking a genuine dependency error (a truly nonexistent
package version would still fail after retries).


