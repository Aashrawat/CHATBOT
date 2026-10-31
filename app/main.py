import asyncio
import json
import time
from pathlib import Path

import openai
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm
from .config import BASE_DIR, settings
from .ingest import ingest_file, prepare_image
from .retrieval import Hit, Index
from .store import Store

store = Store(settings.db_path, settings.chroma_dir)
index = Index(store)

app = FastAPI(title="Doc & Image RAG Chatbot")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

SYSTEM_PROMPT = """You are a precise assistant that answers questions ONLY from the user's own material:
1. SOURCES - numbered excerpts retrieved from the user's uploaded PDFs and images.
2. ATTACHED IMAGES - any image the user attached to their current message.

Rules:
- Ground every factual statement in those materials. Cite excerpts inline as [1], [2] immediately after the \
claim; use separate brackets for multiple sources, e.g. [1][3]. Never cite a number that is not in SOURCES.
- Do NOT use outside knowledge to answer. If the materials do not contain the answer, say so plainly, and \
mention what is missing or which document might help. Never guess or invent details.
- You may reason over, compare, summarise, translate or calculate from the materials.
- If the question is ambiguous, ask one short clarifying question.
- For an attached image, describe or analyse what is actually visible; combine it with SOURCES when relevant.
- Greetings or questions about what you can do need no citations: briefly explain you answer questions about \
uploaded PDFs and images.
- Be clear and well structured. Use Markdown (short lists, tables, bold key terms) when it helps. Reply in the \
user's language."""


# --------------------------------------------------------------------------- helpers
def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, llm.MissingApiKey):
        return str(exc)
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota reached. Check your usage/billing, then try again."
    if isinstance(exc, openai.NotFoundError):
        return "A configured OpenAI model was not found. Check CHAT_MODEL / VISION_MODEL / EMBEDDING_MODEL in .env."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach OpenAI. Check your internet connection."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI error ({exc.status_code}): {getattr(exc, 'message', exc)}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Something went wrong: {exc}"


def build_context(hits: list[Hit]) -> str:
    if not hits:
        return "(no relevant excerpts were found in the uploaded documents)"
    blocks = []
    for n, h in enumerate(hits, 1):
        where = f"{h.doc_name}, page {h.page}" if h.kind == "pdf" else f"{h.doc_name} (image)"
        blocks.append(f"[{n}] ({where})\n{h.text}")
    return "\n\n".join(blocks)


def public_doc(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "hash"}


# --------------------------------------------------------------------------- routes
@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "api_key_configured": bool(settings.openai_api_key),
        "chat_model": settings.chat_model,
        "documents": len(store.list_documents()),
        "chunks": index.size,
    }


@app.get("/api/documents")
async def list_documents():
    return [public_doc(d) for d in store.list_documents()]


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    results = []
    limit = settings.max_upload_mb * 1024 * 1024
    for f in files:
        name = f.filename or "upload"
        try:
            data = await f.read()
            if len(data) > limit:
                raise ValueError(f"File is larger than {settings.max_upload_mb} MB.")
            doc = await ingest_file(store, index, data, name, f.content_type or "")
            results.append({"ok": True, "name": name, "document": public_doc(doc)})
        except Exception as exc:
            results.append({"ok": False, "name": name, "error": friendly_error(exc)})
    return results


def _remove_documents(ids: list[str]) -> int:
    removed = 0
    for doc_id in ids:
        if not doc_id.isalnum():  # ids are hex; never let a crafted id touch other paths
            continue
        if store.delete_document(doc_id):
            removed += 1
        for path in settings.files_dir.glob(f"{doc_id}.*"):
            path.unlink(missing_ok=True)
    return removed


class DeleteRequest(BaseModel):
    ids: list[str]


@app.post("/api/documents/delete")
async def delete_documents(req: DeleteRequest):
    """Remove several documents at once (used when a chat is cleared or the page is closed)."""
    removed = await asyncio.to_thread(_remove_documents, req.ids)
    await asyncio.to_thread(index.reload)
    return {"ok": True, "removed": removed}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not await asyncio.to_thread(_remove_documents, [doc_id]):
        raise HTTPException(404, "Document not found")
    await asyncio.to_thread(index.reload)
    return {"ok": True}


@app.get("/api/documents/{doc_id}/file")
async def document_file(doc_id: str):
    doc = store.get_document(doc_id)
    matches = list(settings.files_dir.glob(f"{doc_id}.*"))
    if not doc or not matches:
        raise HTTPException(404, "File not found")
    return FileResponse(
        matches[0], media_type=doc["mime"], headers={"Content-Disposition": f'inline; filename="{doc["name"]}"'}
    )


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    history: str = Form("[]"),
    doc_ids: str = Form("null"),
    remember: bool = Form(True),
    images: list[UploadFile] = File(default=[]),
):
    message = message.strip()
    if not message:
        raise HTTPException(400, "Message is empty")
    try:
        past = [
            {"role": m["role"], "content": str(m["content"])}
            for m in json.loads(history)
            if m.get("role") in ("user", "assistant") and m.get("content")
        ][-10:]
        scope = json.loads(doc_ids)
    except Exception:
        raise HTTPException(400, "Invalid history or doc_ids")

    # Read uploads now: the request body is gone once streaming starts.
    raw_images: list[tuple[str, bytes]] = []
    try:
        for up in images[:4]:
            data = await up.read()
            if data:
                raw_images.append((up.filename or "photo.jpg", data))
        prepared = [(name, await asyncio.to_thread(prepare_image, data), data) for name, data in raw_images]
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    jpegs = [p[1] for p in prepared]

    async def stream():
        remember_tasks: list[asyncio.Task] = []
        try:
            if remember:
                for name, _jpeg, original in prepared:
                    stamp = time.strftime("%Y-%m-%d %H:%M")
                    label = name if not name.lower().startswith(("camera-", "image.", "blob")) else f"Photo {stamp}.jpg"
                    remember_tasks.append(asyncio.create_task(ingest_file(store, index, original, label, "image/jpeg")))

            hits: list[Hit] = []
            if index.size and scope != []:
                yield sse("status", {"text": "Searching your documents..."})
                query = await llm.rewrite_query(past, message, jpegs)
                qvec = (await llm.embed([query]))[0]
                hits = await asyncio.to_thread(index.search, qvec, query, settings.top_k, scope)
                yield sse("query", {"text": query})

            yield sse(
                "sources",
                [
                    {
                        "n": n,
                        "doc_id": h.doc_id,
                        "name": h.doc_name,
                        "kind": h.kind,
                        "page": h.page,
                        "score": round(h.score, 3),
                        "text": h.text,
                    }
                    for n, h in enumerate(hits, 1)
                ],
            )
            yield sse("status", {"text": "Reading and answering..."})

            user_text = f"SOURCES:\n{build_context(hits)}\n\nQUESTION: {message}"
            if jpegs:
                user_text += f"\n\n(The user attached {len(jpegs)} image(s) to this message; they are shown below.)"
            content: list[dict] = [{"type": "text", "text": user_text}]
            content += [llm.image_part(j) for j in jpegs]
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *past, {"role": "user", "content": content}]

            async for delta in llm.stream_answer(messages):
                yield sse("delta", {"text": delta})

            saved = []
            for task in remember_tasks:
                try:
                    saved.append(public_doc(await task))
                except Exception as exc:
                    saved.append({"error": friendly_error(exc)})
            yield sse("done", {"saved": saved})
        except Exception as exc:
            for task in remember_tasks:
                task.cancel()
            yield sse("error", {"message": friendly_error(exc)})

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
