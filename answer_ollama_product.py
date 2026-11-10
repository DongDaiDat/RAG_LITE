import os, requests
from typing import List

def _build_prompt(question: str, contexts: List[str], citations: List[str]) -> str:
    titles = "\n".join([f"- [Nguồn {i+1}] {citations[i]}" for i in range(len(citations))])
    sources = "\n\n".join([f"[Nguồn {i+1}] {citations[i]}\n{contexts[i]}" for i in range(len(contexts))])

    guide = (
        "Bạn là trợ lý quản lý kho/sản phẩm. Chỉ dùng thông tin trong NGUỒN để trả lời.\n"
        "Quy tắc:\n"
        "1) Tìm SKU/Tên/Model/Danh mục xuất hiện trong TIÊU ĐỀ NGUỒN.\n"
        "2) Nếu câu hỏi về Tồn kho/Giá/Mô tả/Thông số của 1 SKU cụ thể, trích đúng giá trị trong NGUỒN.\n"
        "3) Nếu thiếu căn cứ, trả lời: 'Chưa đủ thông tin trong nguồn'.\n"
        "Luôn thêm phần Trích dẫn dạng [Nguồn i]."
    )

    prompt = f"""{guide}

[CÂU HỎI]
{question}

[TIÊU ĐỀ NGUỒN]
{titles}

[TOÀN VĂN NGUỒN]
{sources}

[ĐỊNH DẠNG TRẢ LỜI]
- Câu trả lời: <ngắn gọn, chính xác theo NGUỒN>
- Trích dẫn: [Nguồn i], ...
"""
    return prompt

def answer_with_ollama_product(question: str, contexts: List[str], citations: List[str]) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    prompt = _build_prompt(question, contexts, citations)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "top_p": 0.9}
    }
    r = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()
