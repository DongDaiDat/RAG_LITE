import re, pdfplumber
from typing import List, Tuple, Dict

ARTICLE_RE = re.compile(r'(?=^\s*Điều\s+\d+\.)', flags=re.M)

def _split_by_article(full_text: str) -> List[Tuple[str,str]]:
    parts = re.split(ARTICLE_RE, full_text)
    out: List[Tuple[str,str]] = []
    for p in parts:
        t = p.strip()
        if not t: 
            continue
        first = t.splitlines()[0].strip()
        title = first if first.lower().startswith("điều") else "Mục/Khác"
        out.append((title, t))
    return out

def _fallback_chunks(text: str, max_chars=1200, overlap=200) -> List[str]:
    out = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j].strip())
        if j == n: break
        i = max(0, j - overlap)
    return [c for c in out if c]

def load_and_chunk_pdf(pdf_path: str, prefer_article=True, min_article=5) -> Tuple[List[str], List[Dict]]:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    full = "\n".join(pages)

    texts: List[str] = []
    metas: List[Dict] = []

    if prefer_article:
        arts = _split_by_article(full)
        if len(arts) >= min_article:
            for title, t in arts:
                texts.append(t)
                metas.append({"title": title, "source": pdf_path})
            return texts, metas

    chunks = _fallback_chunks(full)
    for i, c in enumerate(chunks, 1):
        texts.append(c)
        metas.append({"title": f"Chunk #{i}", "source": pdf_path})
    return texts, metas
