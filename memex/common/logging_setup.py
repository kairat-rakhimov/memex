"""Per-KG-build root logger setup.

Writes ``kg/<id>/build.log`` at DEBUG and mirrors INFO+ to stderr.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(shortname)s  %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_PREFIX = "memex.kg_intelligence."


class _ShortNameFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        record.shortname = name[len(_PREFIX):] if name.startswith(_PREFIX) else name  # type: ignore[attr-defined]
        return super().format(record)


def configure_kg_build_logging(
    log_path: Path,
    *,
    file_level: int = logging.DEBUG,
    stream_level: int = logging.INFO,
) -> logging.FileHandler:
    """Idempotent: re-invocation removes prior handlers this function
    installed, so no duplicate log lines."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_memex_kg_build", False):
            root.removeHandler(h)
            h.close()

    formatter = _ShortNameFormatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler._memex_kg_build = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    stream_handler._memex_kg_build = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    root.setLevel(min(file_level, stream_level))

    # Pin chatty third-party loggers to WARNING (retries/429/5xx still
    # surface; the steady-state success stream goes away).
    for noisy in ("httpx", "httpcore", "urllib3", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # pdfminer emits 100k+ DEBUG lines per PDF parse.
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)

    return file_handler
