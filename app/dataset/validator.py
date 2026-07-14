from pathlib import Path
from app.models.image_record import ImageRecord
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

            if reasons:
                r["_reasons"] = reasons
                rejected.append(r)
            else:
                valid.append(r)

        return valid, rejected