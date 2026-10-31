# DocuLens - RAG chatbot for PDFs and images

Chat with your own PDFs and photos. Answers come only from what you upload, with clickable citations.

## Run

```powershell
uv pip install -r requirements.txt     # once
copy .env               # then put your OPENAI_API_KEY in .env
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## How it works (RAG pipeline)

| Stage | What happens | File |
|---|---|---|
| Ingest PDF | Text per page via PyMuPDF. Scanned pages, and pages whose content is mostly a picture/chart, are rendered and read by GPT vision | `app/ingest.py` |
| Ingest image | EXIF-rotate, downscale, GPT vision writes a description and transcribes all text (OCR) | `app/ingest.py` |
| Chunk + embed | Sentence-aware chunks with overlap, embedded with `text-embedding-3-small`, stored in **ChromaDB** (`data/chroma`) | `app/ingest.py`, `app/store.py` |
| Rewrite | Follow-ups ("what about its price?") become standalone queries; attached photos contribute details | `app/llm.py` |
| Retrieve | ChromaDB cosine vector search + BM25 keyword search, merged with Reciprocal Rank Fusion, de-duplicated with MMR | `app/retrieval.py` |
| Answer | Streamed, grounded in the sources, `[n]` citations, refuses when the material has no answer. Attached photos go straight to the vision model too | `app/main.py` |

## Using it

Everything happens in the message box - there is no separate library.

- Tap **+** and choose **Camera**, **Photos** (gallery) or **Files** (PDFs). Attached files appear above the
  message box; remove one with the x before sending. You can also paste or drag files into the page.
- When you send, PDFs are read and indexed, photos are shown to the vision model, and the answer cites its sources.
- Files stay available for follow-up questions for the rest of that chat. **New chat** (or closing the page) removes them.
- **Camera** opens the live camera (front/back switch). On phones served over plain http the browser blocks live
  camera, so it falls back to the phone's native camera app automatically.

## Config

All optional, see `.env.example` (models, `TOP_K`, chunk size, upload limit).
