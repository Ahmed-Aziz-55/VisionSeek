import logging
import uuid
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Resizes images and assigns UUID-based processed_path. Fails loud if too
    many images break.

    IMPORTANT: This class does NOT normalize pixel values. Normalization
    (mean/std subtraction for CLIP) must happen at embedding-generation time,
    directly on the tensor fed to the model — never baked into a saved JPEG.
    Saving normalized float values as an 8-bit JPEG is lossy and, worse,
    causes CLIP's own processor to normalize an already-normalized image at
    inference time (double transformation), producing incorrect embeddings.

    Use CLIPProcessor (from `transformers`) at embedding time to handle
    resize/crop/normalize consistently with how CLIP was trained:

        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=pil_image, return_tensors="pt")

    This class exists to (a) catch corrupt/unreadable images early, and
    (b) optionally shrink images ahead of time to save disk space and
    speed up later loading — not to replace CLIP's own preprocessing.
    """

    def __init__(
        self,
        output_dir: str,
        target_size: tuple[int, int] = (224, 224),
        failure_threshold: float = 0.1,
        preserve_aspect_ratio: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.failure_threshold = failure_threshold
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def resize_with_padding(self, img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """
        Resize while preserving aspect ratio, padding with black to fill
        the remaining space. Avoids stretching the image content.
        """
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        orig_ratio = orig_w / orig_h
        target_ratio = target_w / target_h

        if orig_ratio > target_ratio:
            new_w = target_w
            new_h = int(target_w / orig_ratio)
        else:
            new_h = target_h
            new_w = int(target_h * orig_ratio)

        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        new_img = Image.new("RGB", target_size, (0, 0, 0))
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        new_img.paste(img_resized, (x_offset, y_offset))

        return new_img

    def simple_resize(self, img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """Resize directly to target size, without preserving aspect ratio (stretches)."""
        return img.resize(target_size, Image.Resampling.LANCZOS)

    def process_image(self, r: dict, base_dir: Path) -> tuple[dict, str | None]:
        """Process a single image record. Returns (record, error_message)."""
        try:
            src_path = base_dir / r["image_path"]

            if not src_path.exists():
                return r, f"image not found: {src_path}"

            img = Image.open(src_path).convert("RGB")

            if self.preserve_aspect_ratio:
                img = self.resize_with_padding(img, self.target_size)
            else:
                img = self.simple_resize(img, self.target_size)

            new_filename = f"{uuid.uuid4().hex}.jpg"
            dest_path = self.output_dir / new_filename
            img.save(dest_path, "JPEG", quality=95, optimize=True)

            r["processed_path"] = str(dest_path)
            return r, None

        except Exception as e:
            return r, f"preprocessing failed: {e}"

    def process(self, records: list[dict], base_image_dir: str = "") -> tuple[list[dict], list[dict]]:
        """Process all records with per-image error handling and threshold checking."""
        base_dir = Path(base_image_dir) if base_image_dir else Path("")
        processed, failed = [], []

        logger.info(f"Processing {len(records)} images...")
        logger.info(f"  Target size: {self.target_size}")
        logger.info(f"  Preserve aspect ratio: {self.preserve_aspect_ratio}")

        for idx, r in enumerate(records):
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Progress: {idx + 1}/{len(records)}")

            processed_record, error = self.process_image(r, base_dir)

            if error:
                r["_reasons"] = r.get("_reasons", []) + [error]
                failed.append(r)
                logger.warning(f"  Failed: {r.get('image_path')} — {error}")
            else:
                processed.append(processed_record)

        failure_rate = len(failed) / len(records) if records else 0

        logger.info("Preprocessing complete:")
        logger.info(f"  Success: {len(processed)}")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Failure rate: {failure_rate:.2%}")

        if failure_rate > self.failure_threshold:
            logger.error(
                f"Preprocessing failure rate {failure_rate:.1%} exceeds threshold "
                f"{self.failure_threshold:.1%} ({len(failed)}/{len(records)} records failed)."
            )
            raise RuntimeError(
                f"Preprocessing failure rate {failure_rate:.1%} exceeds threshold "
                f"{self.failure_threshold:.1%} ({len(failed)}/{len(records)} records failed). "
                f"Stopping pipeline — investigate before continuing."
            )

        return processed, failed

    def get_statistics(self) -> dict:
        """Statistics about images already written to output_dir."""
        all_files = list(self.output_dir.glob("*.jpg"))
        total_size = sum(f.stat().st_size for f in all_files) / (1024 * 1024)  # MB

        return {
            "total_images": len(all_files),
            "total_size_mb": round(total_size, 2),
            "avg_size_kb": round((total_size * 1024) / len(all_files), 2) if all_files else 0,
            "output_dir": str(self.output_dir),
        }