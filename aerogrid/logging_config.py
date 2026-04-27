"""Centralised logging configuration for AeroGrid.

Call :func:`setup_logging` once at the entry point (digital-twin CLI, scripts)
before any other aerogrid imports that may emit log records.  Every module in
the package obtains its own logger via ``logging.getLogger(__name__)``; all
handlers are registered on the *root* logger so a single call here is enough.

Usage::

    from aerogrid.logging_config import setup_logging
    setup_logging(level=logging.DEBUG, console=True)

Format::

    2026-04-28 14:32:01,234 - aerogrid.optimizer - INFO - ...
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    *,
    console: bool = True,
    auto_file: bool = True,
    log_dir: Path | None = None,
) -> Path | None:
    """Configure root logger with file and optional console handlers.

    Args:
        level: Logging level (e.g. ``logging.DEBUG`` or ``logging.INFO``).
            ``DEBUG`` produces very verbose per-sample output; ``INFO`` gives
            high-level events (replans, decisions, solver status).
        log_file: Explicit path for the log file.  When ``None`` and
            ``auto_file`` is ``True`` a timestamped file is created inside
            ``log_dir`` (default ``data/cache/``).
        console: When ``True`` a :class:`~logging.StreamHandler` writing to
            ``stdout`` is added alongside the file handler.
        auto_file: When ``True`` and ``log_file`` is ``None``, automatically
            create a timestamped log file.
        log_dir: Directory for auto-generated log files.  Defaults to
            ``data/cache/`` relative to the repo root.

    Returns:
        The resolved :class:`~pathlib.Path` of the log file, or ``None`` if
        file logging was disabled.
    """
    handlers: list[logging.Handler] = []
    resolved_path: Path | None = None

    if log_file is not None or auto_file:
        if log_file is None:
            if log_dir is None:
                # Lazy import to avoid circular deps at module-import time.
                from aerogrid.config import CACHE_DIR  # noqa: PLC0415
                log_dir = CACHE_DIR
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_file = log_dir / f"aerogrid_{ts}.log"
        resolved_path = Path(log_file)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return resolved_path


__all__ = ["setup_logging"]
