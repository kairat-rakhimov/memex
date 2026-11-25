"""Environment-driven runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_RUN_TIMEOUT_SECONDS = 1800


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    chat_model: str
    reasoning_model: str
    embedding_model: str
    embedding_dimension: int
    host: str
    port: int
    run_timeout_seconds: int
    log_dir: Path
    log_level: str
    sample_mode: bool
    contact: str | None
    extract_mode: str = "session"
    """``session`` (default — fork-per-chunk with the whole doc as a cached prefix:
    maximizes prompt-cache hits and captures cross-section relations) |
    ``chunked`` (stateless per-chunk; the context-ceiling fallback for very large
    docs)."""
    kg_dir: Path = field(default_factory=lambda: Path("kg"))

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            base_url=(os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip(),
            chat_model=os.environ.get("MEMEX_CHAT_MODEL", "gpt-4.1"),
            reasoning_model=os.environ.get("MEMEX_REASONING_MODEL", "gpt-5.1"),
            embedding_model=os.environ.get("MEMEX_EMBEDDING_MODEL", "text-embedding-3-large"),
            embedding_dimension=_int_env("MEMEX_EMBEDDING_DIMENSION", 3072),
            host=os.environ.get("HOST", "127.0.0.1"),
            port=_int_env("PORT", 8000),
            run_timeout_seconds=max(60, _int_env("RUN_TIMEOUT_SECONDS", DEFAULT_RUN_TIMEOUT_SECONDS)),
            log_dir=Path(os.environ.get("LOG_DIR", "logs")).resolve(),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            sample_mode=_bool_env("SAMPLE_MODE", default=False),
            contact=(os.environ.get("MEMEX_CONTACT") or None),
            extract_mode=(os.environ.get("MEMEX_EXTRACT_MODE") or "session").strip().lower(),
            kg_dir=Path(os.environ.get("MEMEX_KG_DIR", "kg")).resolve(),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url)
