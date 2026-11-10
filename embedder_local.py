import os
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
def get_embedder():
    global _model
    if _model is None:
        name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        _model = SentenceTransformer(name)
    return _model

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")
