import logging
from pathlib import Path
from pydantic import ValidationError

from app.models.image_record import ImageRecord

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Rule-based pass/fail. Splits records into valid vs rejected."""

    def __init__(self, records: list[dict], base_image_dir: str = ""):
        self.records = records
        self.base_image_dir = Path(base_image_dir)

    def validate(self) -> tuple[list[dict], list[dict]]:
        valid, rejected = [], []

        for r in self.records:
            if "_error" in r:
                rejected.append(r)
                continue

            reasons = []
            if not r["caption"].strip():
                reasons.append("empty caption")
            if not r["category"].strip():
                reasons.append("empty category")

            full_path = self.base_image_dir / r["image_path"]
            if not full_path.exists():
                reasons.append(f"image not found: {full_path}")

            # Runtime schema validation via Pydantic, on top of the manual
            # checks above. This is where ImageRecord is actually exercised
            # (not just imported) — it catches shape/type issues the manual
            # checks don't (e.g. a non-string field from a malformed
            # source), and its field_validator independently re-checks for
            # whitespace-only strings.
            if not reasons:
                try:
                    ImageRecord(
                        image_path=r["image_path"],
                        caption=r["caption"],
                        category=r["category"],
                    )
                except ValidationError as e:
                    reasons.append(f"schema validation failed: {e}")

            if reasons:
                r["_reasons"] = reasons
                rejected.append(r)
            else:
                valid.append(r)

        return valid, rejected