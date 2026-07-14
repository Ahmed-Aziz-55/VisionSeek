from collections import Counter
from app.models.image_record import ImageRecord
class DatasetInspector:
    """Pure descriptive stats — no filtering, no rejection."""

    def __init__(self, records: list[dict]):
        self.records = [r for r in records if "_error" not in r]

    def summary(self) -> dict:
        captions = [r["caption"] for r in self.records]
        categories = [r["category"] for r in self.records]

        return {
            "total_rows": len(self.records),
            "unique_categories": len(set(categories)),
            "category_distribution": dict(Counter(categories)),
            "avg_caption_length": (
                sum(len(c.split()) for c in captions) / len(captions)
                if captions else 0
            ),
        }