import json

from app.services.embedding_generator import EmbeddingGenerator

TEST_MODE = False
TEST_SIZE = 100

with open("data/processed/manifest.json") as f:
    records = json.load(f)

print(f"Loaded {len(records)} preprocessed records from manifest\n")

if TEST_MODE:
    records = records[:TEST_SIZE]
    print(f"TEST MODE: using only {len(records)} records\n")

generator = EmbeddingGenerator(batch_size=64)

print("Generating image embeddings...")
img_emb, img_records = generator.generate_image_embeddings(records)
EmbeddingGenerator.save(img_emb, img_records, "embeddings", "image")

print("\nGenerating text embeddings...")
txt_emb, txt_records = generator.generate_text_embeddings(records)
EmbeddingGenerator.save(txt_emb, txt_records, "embeddings", "text")

print(f"\nDone. Image embeddings shape: {img_emb.shape}, Text embeddings shape: {txt_emb.shape}")