import os
from typing import List, Dict
import faiss, numpy as np
from FlagEmbedding import FlagReranker
from .embedder_local import embed_texts, get_embedder
from .index_faiss import load_index_bundle

class Retriever:
    def __init__(self, index_dir: str, embedder=None):
        self.index_dir = index_dir
        self.embedder = embedder or get_embedder()
        self.index, self.metas, self.texts = load_index_bundle(index_dir)
        name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.reranker = FlagReranker(name, use_fp16=False)

    def search(self, query: str, top_k=30) -> List[Dict]:
        qv = embed_texts(self.embedder, [query])
        D, I = self.index.search(qv, top_k)
        I, D = I[0].tolist(), D[0].tolist()
        out = []
        for idx, score in zip(I, D):
            out.append({"index": idx, "score": float(score), "meta": self.metas[idx], "text": self.texts[idx]})
        return out

    def rerank(self, query: str, candidates: List[Dict], top_n=6) -> List[Dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.compute_score(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
