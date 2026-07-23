import pytest
from app.dataset.validator import DatasetValidator


@pytest.fixture
def existing_image(tmp_path):
    """A real file on disk, so the 'image exists' check can pass."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "real.jpg"
    img_path.write_bytes(b"fake image bytes")
    return tmp_path, "images/real.jpg"


def test_valid_record_passes(existing_image):
    base_dir, image_path = existing_image
    records = [{"image_path": image_path, "caption": "a caption", "category": "image"}]

    valid, rejected = DatasetValidator(records, base_image_dir=str(base_dir)).validate()

    assert len(valid) == 1
    assert len(rejected) == 0


def test_loader_error_rejected_immediately():
    records = [{"_error": "line 3: expected 3 fields, got 2"}]

    valid, rejected = DatasetValidator(records, base_image_dir="").validate()

    assert len(valid) == 0
    assert len(rejected) == 1
    assert rejected[0] == records[0]


def test_empty_caption_rejected(existing_image):
    base_dir, image_path = existing_image
    records = [{"image_path": image_path, "caption": "   ", "category": "image"}]

    valid, rejected = DatasetValidator(records, base_image_dir=str(base_dir)).validate()

    assert len(valid) == 0
    assert "empty caption" in rejected[0]["_reasons"]


def test_empty_category_rejected(existing_image):
    base_dir, image_path = existing_image
    records = [{"image_path": image_path, "caption": "a caption", "category": ""}]

    valid, rejected = DatasetValidator(records, base_image_dir=str(base_dir)).validate()

    assert len(valid) == 0
    assert "empty category" in rejected[0]["_reasons"]


def test_missing_image_rejected(tmp_path):
    records = [{"image_path": "does_not_exist.jpg", "caption": "a caption", "category": "image"}]

    valid, rejected = DatasetValidator(records, base_image_dir=str(tmp_path)).validate()

    assert len(valid) == 0
    assert any("image not found" in r for r in rejected[0]["_reasons"])


def test_multiple_failure_reasons_all_captured(tmp_path):
    """A record can fail more than one rule at once — all reasons should show up."""
    records = [{"image_path": "missing.jpg", "caption": "", "category": ""}]

    valid, rejected = DatasetValidator(records, base_image_dir=str(tmp_path)).validate()

    assert len(valid) == 0
    reasons = rejected[0]["_reasons"]
    assert "empty caption" in reasons
    assert "empty category" in reasons
    assert any("image not found" in r for r in reasons)