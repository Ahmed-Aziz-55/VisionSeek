import numpy as np
from app.index.index_builder import IndexBuilder

img_emb = np.load("embeddings/image_embeddings.npy")

builder = IndexBuilder(dimension=img_emb.shape[1])
index = builder.build(img_emb)

print("Total vectors in index:", index.ntotal)

builder.save("index/image_index.faiss")
print("Index saved to index/image_index.faiss")