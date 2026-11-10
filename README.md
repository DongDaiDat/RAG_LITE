# RAG-Lite: Local & High-Precision Retrieval-Augmented Generation

**RAG-Lite** là thư viện/hệ thống RAG (Retrieval-Augmented Generation) mô-đun hóa, nhẹ và tối ưu hóa cao cho việc tra cứu thông tin chính xác từ văn bản phức tạp (tài liệu pháp lý, hợp đồng, cơ sở dữ liệu sản phẩm, quản lý kho hàng).

Hệ thống hoạt động **100% Offline/Cục bộ (Local)**, giúp bảo mật dữ liệu tuyệt đối, không phát sinh chi phí API và được thiết kế theo kiến trúc **Two-Stage Retrieval (Bi-Encoder + Cross-Encoder)** giúp triệt tiêu nhiễu và hạn chế tối đa hiện tượng bịa thông tin (Hallucination) từ LLM.

---

## Tính năng nổi bật

- **Chunking thông minh theo cấu trúc (`splitter.py`):** Phân đoạn văn bản theo cấu trúc Điều/Mục (Article-based) chuyên biệt cho văn bản pháp luật bằng Regex, kết hợp giải thuật dự phòng (Fallback Overlapping Chunks) giữ nguyên ngữ cảnh.
- **Tìm kiếm Vector siêu tốc (`index_faiss.py` & `embedder_local.py`):** Sử dụng mô hình nhúng `BAAI/bge-m3` chuẩn hóa vector cosine kết hợp chỉ mục `FAISS` (IndexFlatIP) để lọc ra các đoạn văn bản liên quan.
- **Đánh giá lại điểm số (Rerank Engine - `retrieval.py`):** Sử dụng Cross-Encoder `BAAI/bge-reranker-v2-m3` để xếp hạng lại Top-K kết quả, đảm bảo đoạn văn bản được đưa vào LLM là tương quan nhất.
- **Kiểm soát thông tin nghiêm ngặt (`answer_ollama.py` & `answer_ollama_product.py`):** Prompt Engineering tối ưu cho Local LLM (`llama3.1:8b` qua Ollama), ép mô hình chỉ trả lời dựa trên ngữ cảnh được cung cấp, tự động bổ sung trích dẫn nguồn `[Nguồn i]` và phản hồi *"Chưa đủ thông tin trong nguồn"* nếu dữ liệu thiếu.

---

## Kiến trúc hệ thống

```
                                 ┌───────────────────────┐
                                 │   Tài liệu PDF / Data  │
                                 └──────────┬────────────┘
                                            │ Chunking (Article / Fallback)
                                            ▼
                                 ┌───────────────────────┐
                                 │ Metadata & Texts Json │
                                 └──────────┬────────────┘
                                            │ Embedding (BGE-M3)
                                            ▼
                                 ┌───────────────────────┐
                                 │  FAISS Vector Index   │
                                 └──────────┬────────────┘
                                            │
[User Question] ───────────────► Giai đoạn 1: FAISS Search (Top-K)
                                            │
                                            ▼
                                 Giai đoạn 2: BGE Reranker (Top-N)
                                            │
                                            ▼
                                 Prompt Engineering (Context + Strict Rules)
                                            │
                                            ▼
                                 Ollama (Llama 3.1: 8B Local)
                                            │
                                            ▼
                                 [Câu trả lời + Trích dẫn nguồn]
```

---

## Yêu cầu hệ thống & Cài đặt

### 1. Yêu cầu tiền đề
- **Python:** >= 3.10
- **Ollama:** Đã cài đặt [Ollama](https://ollama.ai/) và kéo model `llama3.1:8b` về máy:
  ```bash
  ollama pull llama3.1:8b
  ```

### 2. Cài đặt môi trường
Cài đặt các phụ thuộc cần thiết bằng `pip`:

```bash
pip install faiss-cpu sentence-transformers FlagEmbedding pdfplumber numpy requests
```

---

## Hướng dẫn sử dụng

### 1. Phân đoạn & Đóng chỉ mục tài liệu (Indexing)
```python
from rag_lite.splitter import load_and_chunk_pdf
from rag_lite.embedder_local import get_embedder, embed_texts
from rag_lite.index_faiss import save_index_bundle

# 1. Trích xuất & Chunking PDF
texts, metas = load_and_chunk_pdf("duong_dan_tai_lieu.pdf")

# 2. Nhúng Vector
embedder = get_embedder()
vectors = embed_texts(embedder, texts)

# 3. Lưu vào FAISS Index
save_index_bundle(vectors, texts, metas, out_dir="./storage_index")
```

### 2. Truy vấn & Trả lời câu hỏi (Querying)
```python
from rag_lite.retrieval import Retriever
from rag_lite.answer_ollama import answer_with_ollama

# 1. Khởi tạo Retriever
retriever = Retriever(index_dir="./storage_index")

question = "Khi nào người sử dụng lao động bị xử lý kỷ luật?"

# 2. Search & Rerank 2 giai đoạn
candidates = retriever.search(question, top_k=30)
reranked_docs = retriever.rerank(question, candidates, top_n=5)

contexts = [doc["text"] for doc in reranked_docs]
citations = [doc["meta"]["title"] for doc in reranked_docs]

# 3. Sinh câu trả lời từ Ollama LLM
response = answer_with_ollama(question, contexts, citations)
print(response)
```

---

## Biến môi trường (Environment Variables)

Bạn có thể tùy chỉnh mô hình nhúng, reranker hoặc Ollama endpoint thông qua biến môi trường:

| Biến môi trường | Giá trị mặc định | Mô tả |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Mô hình nhúng văn bản local |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Mô hình xếp hạng lại tương quan |
| `OLLAMA_HOST` | `http://localhost:11434` | Endpoint của Ollama service |
| `OLLAMA_MODEL` | `llama3.1:8b` | Tên mô hình LLM chạy trên Ollama |

---
