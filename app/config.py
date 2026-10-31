import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", ""))

    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gpt-4o"))
    fast_model: str = field(default_factory=lambda: _env("FAST_MODEL", "gpt-4o-mini"))
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "gpt-4o"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))

    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "8")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "200")))

    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "50")))
    max_vision_pages: int = field(default_factory=lambda: int(_env("MAX_VISION_PAGES", "40")))
    max_image_side: int = 2048

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BASE_DIR / "data"))))

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chatbot.db"


settings = Settings()
settings.files_dir.mkdir(parents=True, exist_ok=True)
