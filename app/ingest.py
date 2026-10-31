"""Turn uploaded PDFs / images into searchable, embedded chunks."""
import asyncio
import hashlib
import io
import re
import uuid
from dataclasses import dataclass

import pymupdf
from PIL import Image, ImageOps

from . import llm
from .config import settings
from .retrieval import Index
from .store import Store

MIN_TEXT_CHARS = 40  # below this a PDF page is treated as scanned
SHORT_PAGE_CHARS = 400  # pages this short with a big picture get the picture described too


# --------------------------------------------------------------------------- helpers
def prepare_image(data: bytes) -> bytes:
    """Decode any image, fix EXIF rotation, downscale, and return JPEG bytes."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except Exception as exc:
        raise ValueError("That file is not a readable image.") from exc
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, "white")
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    img.thumbnail((settings.max_image_side, settings.max_image_side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88, optimize=True)
    return out.getvalue()


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line wraps
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _units(text: str, size: int):
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= size:
            yield line
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if len(sentence) <= size:
                yield sentence
            else:
                for i in range(0, len(sentence), size):
                    yield sentence[i : i + size]


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Greedy line/sentence packing with overlap, so chunks never cut mid-sentence."""
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in _units(text, size):
        if current and length + len(unit) + 1 > size:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(unit)
        length += len(unit) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# --------------------------------------------------------------------------- PDF
@dataclass
class PageData:
    number: int
    text: str
    image: bytes | None = None  # rendered page, only when vision is needed
    mode: str | None = None  # "replace" (scanned) or "augment" (text + figure)


def _has_big_image(page: "pymupdf.Page") -> bool:
    area = page.rect.width * page.rect.height
    if area <= 0:
        return False
    try:
        for info in page.get_image_info():
            rect = pymupdf.Rect(info["bbox"])
            if rect.width * rect.height > 0.25 * area:
                return True
    except Exception:
        pass
    return False


def extract_pdf(data: bytes) -> list[PageData]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("That PDF could not be opened (it may be corrupted).") from exc
    if doc.needs_pass:
        raise ValueError("That PDF is password-protected.")

    pages: list[PageData] = []
    budget = settings.max_vision_pages
    for i, page in enumerate(doc):
        text = _clean(page.get_text("text", sort=True))
        mode = None
        if len(text) < MIN_TEXT_CHARS:
            mode = "replace"
        elif len(text) < SHORT_PAGE_CHARS and _has_big_image(page):
            mode = "augment"
        image = None
        if mode and budget > 0:
            image = page.get_pixmap(dpi=150).tobytes("jpeg")
            budget -= 1
        pages.append(PageData(i + 1, text, image, mode))
    doc.close()
    return pages


async def _pdf_chunks(data: bytes) -> tuple[list[tuple[int, str]], int]:
    pages = await asyncio.to_thread(extract_pdf, data)
    sem = asyncio.Semaphore(4)

    async def read_visually(p: PageData) -> None:
        async with sem:
            try:
                vision = await llm.transcribe_page(p.image)
            except llm.MissingApiKey:
                raise
            except Exception:
                return  # keep whatever text layer we have
        if p.mode == "replace":
            p.text = vision
        elif vision:
            p.text = f"{p.text}\n\n[Visual content on this page]\n{vision}"

    await asyncio.gather(*(read_visually(p) for p in pages if p.image))

    chunks: list[tuple[int, str]] = []
    for p in pages:
        for piece in chunk_text(p.text):
            chunks.append((p.number, piece))
    return chunks, len(pages)


# --------------------------------------------------------------------------- orchestration
def _sniff_kind(data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return "pdf"
    return "image"


async def ingest_file(store: Store, index: Index, data: bytes, filename: str, mime: str = "") -> dict:
    """Parse, chunk, embed and persist one file. Returns the document record."""
    if not data:
        raise ValueError("The file is empty.")

    digest = hashlib.sha256(data).hexdigest()
    existing = store.find_by_hash(digest)
    if existing:
        return {**existing, "duplicate": True}

    kind = _sniff_kind(data)
    if kind == "pdf":
        chunks, pages = await _pdf_chunks(data)
        stored, ext, out_mime = data, ".pdf", "application/pdf"
    else:
        jpeg = await asyncio.to_thread(prepare_image, data)
        description = await llm.describe_image(jpeg)
        chunks = [(1, c) for c in chunk_text(description, size=max(settings.chunk_size, 1800))]
        pages, stored, ext, out_mime = 1, jpeg, ".jpg", "image/jpeg"

    if not chunks:
        raise ValueError("No readable text was found in that file.")

    embed_inputs = [f"{filename} (page {page})\n{text}"[:7000] for page, text in chunks]
    embeddings = await llm.embed(embed_inputs)

    doc_id = uuid.uuid4().hex[:12]
    await asyncio.to_thread(
        store.add_document,
        doc_id=doc_id,
        name=filename,
        kind=kind,
        mime=out_mime,
        pages=pages,
        size=len(data),
        digest=digest,
        chunks=chunks,
        embeddings=embeddings,
    )
    (settings.files_dir / f"{doc_id}{ext}").write_bytes(stored)
    await asyncio.to_thread(index.reload)
    return {**store.get_document(doc_id), "duplicate": False}
