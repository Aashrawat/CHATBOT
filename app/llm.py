"""Thin async wrapper around the OpenAI API: embeddings, vision, rewriting, streaming."""
import asyncio
import base64
from typing import AsyncIterator

import numpy as np
from openai import AsyncOpenAI

from .config import settings


class MissingApiKey(RuntimeError):
    pass


_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise MissingApiKey("OPENAI_API_KEY is not set. Add it to the .env file and restart the server.")
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=120,
            max_retries=3,
        )
    return _client


def image_part(jpeg: bytes, detail: str = "high") -> dict:
    b64 = base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}


# --------------------------------------------------------------------------- embeddings
async def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    sem = asyncio.Semaphore(4)
    batches = [texts[i : i + 64] for i in range(0, len(texts), 64)]

    async def run(batch: list[str]) -> list[list[float]]:
        async with sem:
            res = await client().embeddings.create(model=settings.embedding_model, input=batch)
            return [d.embedding for d in res.data]

    results = await asyncio.gather(*(run(b) for b in batches))
    arr = np.array([e for r in results for e in r], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# --------------------------------------------------------------------------- vision
IMAGE_INDEX_PROMPT = """You are preparing an image for a search index. Produce exactly two sections.

DESCRIPTION: A thorough, factual description of what the image shows: objects, people, scene, layout, colours, \
and the meaning of any chart, diagram, screenshot or UI. Mention distinctive details someone might search for.

TEXT: Transcribe ALL visible text verbatim, in reading order. Render tables as Markdown tables. \
If there is no text, write "None".

Do not speculate beyond what is visible."""

PAGE_OCR_PROMPT = """Transcribe this document page faithfully and completely, in reading order.
- Keep headings, lists and paragraphs. Render tables as Markdown tables.
- Replace each figure, chart, diagram or photo with a one or two sentence description in the form \
[Figure: ...], including key numbers or trends for charts.
- Output only the page content, with no commentary."""


async def _vision(prompt: str, jpeg: bytes) -> str:
    res = await client().chat.completions.create(
        model=settings.vision_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(jpeg)]}],
    )
    return (res.choices[0].message.content or "").strip()


async def describe_image(jpeg: bytes) -> str:
    return await _vision(IMAGE_INDEX_PROMPT, jpeg)


async def transcribe_page(jpeg: bytes) -> str:
    return await _vision(PAGE_OCR_PROMPT, jpeg)


# --------------------------------------------------------------------------- query rewriting
REWRITE_PROMPT = """Rewrite the user's latest message as a single standalone search query for a document \
search engine.
- Resolve pronouns and references using the conversation.
- If an image is attached, include the details from it that matter for the question (visible text, product \
names, error codes, objects).
- Keep it concise and keyword-rich. Output ONLY the query, no quotes or explanation."""


async def rewrite_query(history: list[dict], question: str, images: list[bytes]) -> str:
    if not history and not images:
        return question
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:800]}" for m in history[-6:])
    text = f"Conversation so far:\n{convo or '(none)'}\n\nLatest user message: {question}"
    if images:
        text += "\n\n(The user attached the image(s) shown below.)"
    content: list[dict] = [{"type": "text", "text": text}]
    content += [image_part(img, "low") for img in images[:3]]
    try:
        res = await client().chat.completions.create(
            model=settings.fast_model,
            messages=[{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": content}],
        )
        out = (res.choices[0].message.content or "").strip().strip('"')
        return out or question
    except MissingApiKey:
        raise
    except Exception:
        return question  # rewriting is an optimisation; never fail the chat because of it


# --------------------------------------------------------------------------- answer generation
async def stream_answer(messages: list[dict]) -> AsyncIterator[str]:
    stream = await client().chat.completions.create(model=settings.chat_model, messages=messages, stream=True)
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
