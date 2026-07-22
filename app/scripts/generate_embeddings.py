import json
import logging

from app.core.logging_config import setup_logging
from app.services.embedding_generator import EmbeddingGenerator

setup_logging()
logger = logging.getLogger(__name__)

TEST_MODE = False
TEST_SIZE = 100

with open("data/processed/manifest.json") as f:
    records = json.load(f)

logger.info(f"Loaded {len(records)} preprocessed records from manifest")

if TEST_MODE:
    records = records[:TEST_SIZE]
    logger.info(f"TEST MODE: using only {len(records)} records")

generator = EmbeddingGenerator(model_name="openai/clip-vit-large-patch14", batch_size=64)

logger.info("Generating image embeddings...")
img_emb, img_records = generator.generate_image_embeddings(records)
EmbeddingGenerator.save(img_emb, img_records, "embeddings", "image")

logger.info("Generating text embeddings...")
txt_emb, txt_records = generator.generate_text_embeddings(records)
EmbeddingGenerator.save(txt_emb, txt_records, "embeddings", "text")

logger.info(f"Done. Image embeddings shape: {img_emb.shape}, Text embeddings shape: {txt_emb.shape}")