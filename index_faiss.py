import os, json
from typing import List, Dict
import faiss
import numpy as np

def save_index_bundle(X: np.ndarray, texts: List[str], metas: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"[FAISS] dim={dim}, added={len(texts)}")

def load_index_bundle(dir_: str):
    index = faiss.read_index(os.path.join(dir_, "index.faiss"))
    metas = []
    with open(os.path.join(dir_, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    texts = []
    with open(os.path.join(dir_, "texts.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return index, metas, texts
