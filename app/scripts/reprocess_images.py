import json
import logging
from pathlib import Path

from app.core.logging_config import setup_logging
from app.dataset.loader import DatasetLoader
from app.dataset.validator import DatasetValidator
from app.dataset.preprocessor import Preprocessor

setup_logging()
logger = logging.getLogger(__name__)

logger.info("Loading dataset...")
records = DatasetLoader("data/raw/dataset.txt").load()
logger.info(f"Loaded {len(records)} records")

logger.info("Validating...")
validator = DatasetValidator(records, base_image_dir="")
valid_records, rejected_records = validator.validate()
logger.info(f"Valid: {len(valid_records)}, Rejected: {len(rejected_records)}")

logger.info("Preprocessing images...")
preprocessor = Preprocessor(
    output_dir="processed/images",
    target_size=(224, 224),
    failure_threshold=0.1,
    preserve_aspect_ratio=True,
)

processed, failed = preprocessor.process(valid_records, base_image_dir="")

logger.info(f"Final stats: {preprocessor.get_statistics()}")

# Save a manifest so downstream steps (embedding generation) don't need to
# re-run preprocessing — the image_path -> processed_path mapping would
# otherwise be lost once this script exits.
manifest_path = Path("data/processed/manifest.json")
manifest_path.parent.mkdir(parents=True, exist_ok=True)

with open(manifest_path, "w") as f:
    json.dump(processed, f)

logger.info(f"Saved manifest with {len(processed)} records to {manifest_path}")