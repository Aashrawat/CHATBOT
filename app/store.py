"""Persistence: ChromaDB holds chunks + embeddings, SQLite holds the document registry."""
import sqlite3
import threading
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

COLLECTION = "chunks"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    mime        TEXT,
    pages       INTEGER NOT NULL DEFAULT 1,
    size        INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    hash        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
"""

_DOC_COLUMNS = "id, name, kind, mime, pages, size, chunks, created_at"


class Store:
    def __init__(self, sqlite_path: Path, chroma_dir: Path):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

        self._chroma = chromadb.PersistentClient(
            path=str(chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._chroma.get_or_create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )
        self._upgrade_legacy_schema()

    # ------------------------------------------------------------------ legacy upgrade
    def _upgrade_legacy_schema(self) -> None:
        """Older versions kept chunks in SQLite. Move them into Chroma once, then drop the table."""
        with self._lock:
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(documents)")}
            if "chunks" not in cols:
                self._db.execute("ALTER TABLE documents ADD COLUMN chunks INTEGER NOT NULL DEFAULT 0")
            has_legacy = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
            ).fetchone()
            if has_legacy:
                rows = self._db.execute(
                    "SELECT c.id, c.doc_id, d.name, d.kind, c.page, c.text, c.embedding "
                    "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
                ).fetchall()
                by_doc: dict[str, list] = {}
                for r in rows:
                    by_doc.setdefault(r["doc_id"], []).append(r)
                for doc_id, items in by_doc.items():
                    self._chroma_add(
                        doc_id,
                        items[0]["name"],
                        items[0]["kind"],
                        [(r["page"], r["text"]) for r in items],
                        np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in items]),
                    )
                    self._db.execute("UPDATE documents SET chunks = ? WHERE id = ?", (len(items), doc_id))
                self._db.execute("DROP TABLE chunks")
            self._db.commit()

    # ------------------------------------------------------------------ chroma helpers
    def _chroma_add(self, doc_id: str, name: str, kind: str, chunks: list[tuple[int, str]], emb: np.ndarray) -> None:
        batch = max(1, min(self._chroma.get_max_batch_size(), 2000))
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self.collection.upsert(
                ids=[f"{doc_id}:{start + i}" for i in range(len(part))],
                embeddings=emb[start : start + len(part)].astype(np.float32),
                documents=[text for _, text in part],
                metadatas=[{"doc_id": doc_id, "doc_name": name, "kind": kind, "page": page} for page, _ in part],
            )

    def iter_chunks(self) -> list[dict]:
        """All chunks (text + metadata, no vectors) - used to build the keyword index."""
        out: list[dict] = []
        offset, page_size = 0, 5000
        while True:
            res = self.collection.get(limit=page_size, offset=offset, include=["documents", "metadatas"])
            ids = res["ids"]
            if not ids:
                break
            for cid, text, meta in zip(ids, res["documents"], res["metadatas"]):
                out.append(
                    {
                        "id": cid,
                        "doc_id": meta["doc_id"],
                        "doc_name": meta["doc_name"],
                        "kind": meta["kind"],
                        "page": int(meta["page"]),
                        "text": text,
                    }
                )
            offset += len(ids)
        return out

    # ------------------------------------------------------------------ documents
    def find_by_hash(self, digest: str) -> dict | None:
        with self._lock:
            row = self._db.execute(f"SELECT {_DOC_COLUMNS} FROM documents WHERE hash = ?", (digest,)).fetchone()
        return dict(row) if row else None

    def get_document(self, doc_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(f"SELECT {_DOC_COLUMNS} FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(f"SELECT {_DOC_COLUMNS} FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def add_document(
        self,
        *,
        doc_id: str,
        name: str,
        kind: str,
        mime: str,
        pages: int,
        size: int,
        digest: str,
        chunks: list[tuple[int, str]],
        embeddings: np.ndarray,
    ) -> None:
        """Store vectors in Chroma, then register the document; roll back Chroma if registration fails."""
        with self._lock:
            self._chroma_add(doc_id, name, kind, chunks, embeddings)
            try:
                self._db.execute(
                    "INSERT INTO documents (id, name, kind, mime, pages, size, chunks, hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, name, kind, mime, pages, size, len(chunks), digest, time.time()),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                self.collection.delete(where={"doc_id": doc_id})
                raise

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._db.commit()
            self.collection.delete(where={"doc_id": doc_id})
            return cur.rowcount > 0
