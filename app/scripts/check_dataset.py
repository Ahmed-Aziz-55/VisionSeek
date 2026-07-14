from app.dataset.loader import DatasetLoader
from app.dataset.validator import DatasetValidator
from app.dataset.inspector import DatasetInspector

records = DatasetLoader("datasets/images/results.csv").load()

inspector = DatasetInspector(records)
stats = inspector.summary()
print(stats)