import pytest
from app.dataset.loader import DatasetLoader


@pytest.fixture
def dataset_file(tmp_path):
    content = (
        "img1.jpg|a caption for image one|image\n"
        "img2.jpg|a caption for image two|image\n"
        "\n"
        "malformed line with only two fields\n"
        "img3.jpg|caption|image|extra_field\n"
    )
    path = tmp_path / "dataset.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_returns_correct_count(dataset_file):
    records = DatasetLoader(dataset_file).load()
    assert len(records) == 4


def test_well_formed_rows_parsed_correctly(dataset_file):
    records = DatasetLoader(dataset_file).load()
    good = [r for r in records if "_error" not in r]
    assert len(good) == 2


def test_malformed_rows_flagged_not_dropped(dataset_file):
    records = DatasetLoader(dataset_file).load()
    errors = [r for r in records if "_error" in r]
    assert len(errors) == 2


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DatasetLoader("does/not/exist.txt").load()
