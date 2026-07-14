import json
from pathlib import Path

from app.dataset.loader import DatasetLoader
from app.dataset.validator import DatasetValidator
from app.dataset.preprocessor import Preprocessor

print("Loading dataset...")
records = DatasetLoader("data/raw/dataset.txt").load()
print(f"Loaded {len(records)} records\n")

print("Validating...")
validator = DatasetValidator(records, base_image_dir="")
valid_records, rejected_records = validator.validate()
print(f"Valid: {len(valid_records)}, Rejected: {len(rejected_records)}\n")

print("Preprocessing images...")
preprocessor = Preprocessor(
    output_dir="processed/images",
    target_size=(224, 224),
    failure_threshold=0.1,
    preserve_aspect_ratio=True,
)

processed, failed = preprocessor.process(valid_records, base_image_dir="")

print(f"\nFinal stats: {preprocessor.get_statistics()}")

# Save a manifest so downstream steps (embedding generation) don't need to
# re-run preprocessing — the image_path -> processed_path mapping would
# otherwise be lost once this script exits.
manifest_path = Path("data/processed/manifest.json")
manifest_path.parent.mkdir(parents=True, exist_ok=True)

with open(manifest_path, "w") as f:
    json.dump(processed, f)

print(f"Saved manifest with {len(processed)} records to {manifest_path}")