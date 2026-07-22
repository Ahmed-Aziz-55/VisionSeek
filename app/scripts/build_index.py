import logging

import numpy as np

from app.core.logging_config import setup_logging
from app.index.index_builder import IndexBuilder

setup_logging()
logger = logging.getLogger(__name__)

img_emb = np.load("embeddings/image_embeddings.npy")

builder = IndexBuilder(dimension=img_emb.shape[1])
index = builder.build(img_emb)

logger.info(f"Total vectors in index: {index.ntotal}")

builder.save("index/image_index.faiss")
logger.info("Index saved to index/image_index.faiss")