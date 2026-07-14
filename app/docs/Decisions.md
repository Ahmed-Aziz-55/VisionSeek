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

---

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