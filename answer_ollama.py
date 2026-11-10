# rag_lite/answer_ollama.py
import os, requests
from typing import List

def _build_prompt(question: str, contexts: List[str], citations: List[str]) -> str:
    # Liệt kê tiêu đề nguồn cho model "nhìn" thấy trước
    titles = "\n".join([f"- [Nguồn {i+1}] {citations[i]}" for i in range(len(citations))])
    sources = "\n\n".join([f"[Nguồn {i+1}] {citations[i]}\n{contexts[i]}" for i in range(len(contexts))])

    guide = (
        "Bạn là trợ lý pháp lý, CHỈ dùng thông tin trong NGUỒN để trả lời.\n"
        "QUY TRÌNH BẮT BUỘC:\n"
        "1) Liệt kê các Điều/Mục xuất hiện trong TIÊU ĐỀ NGUỒN.\n"
        "2) Nếu câu hỏi chỉ rõ 'Điều X' và TIÊU ĐỀ có 'Điều X', hãy tóm tắt/nêu nội dung chính của Điều X; "
        "có thể trích dẫn ngắn các điểm quan trọng.\n"
        "3) Nếu KHÔNG tìm thấy Điều X trong NGUỒN, trả lời đúng một câu: 'Chưa đủ thông tin trong nguồn'.\n"
        "Luôn thêm phần Trích dẫn dạng [Nguồn i]. Không thêm thông tin ngoài nguồn."
    )

    prompt = f"""{guide}

[CÂU HỎI]
{question}

[TIÊU ĐỀ NGUỒN]
{titles}

[TOÀN VĂN NGUỒN]
{sources}

[ĐỊNH DẠNG TRẢ LỜI]
- Câu trả lời: <nội dung ngắn gọn>
- Trích dẫn: [Nguồn i], ...
"""
    return prompt

def answer_with_ollama(question: str, contexts: List[str], citations: List[str]) -> str:
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

