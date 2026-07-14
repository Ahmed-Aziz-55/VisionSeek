from pathlib import Path

class DatasetLoader:
    """Reads raw pipe-separated dataset file, returns raw dict rows (no validation)."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> list[dict]:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")

        records = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # skip blank lines
                parts = line.split("|")
                if len(parts) != 3:
                    # Malformed schema — loader doesn't decide accept/reject,
                    # it just flags it so Validator can act later.
                    records.append({"_error": f"line {line_num}: expected 3 fields, got {len(parts)}"})
                    continue
                image_path, caption, category = parts
                records.append({
                    "image_path": image_path,
                    "caption": caption,
                    "category": category,
                })
        return records