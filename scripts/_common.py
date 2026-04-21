"""Shared utilities for data fetchers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


class FetchError(RuntimeError):
    pass


def http_get(url: str, timeout: float = 20.0, **kwargs: Any) -> requests.Response:
    """Single attempt GET with a short timeout. Callers handle retries/fallback."""
    return requests.get(url, timeout=timeout, **kwargs)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_manifest(
    out_path: Path,
    *,
    source: str,             # "real" or "synthetic"
    url_base: str | None,
    windows: dict[str, tuple[datetime, datetime]],
    files: dict[str, Path],
    extras: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "url_base": url_base,
        "windows": {
            k: [s.isoformat(), e.isoformat()] for k, (s, e) in windows.items()
        },
        "files": {
            label: {
                "path": str(p.relative_to(out_path.parent)),
                "bytes": p.stat().st_size if p.exists() else 0,
                "sha256": sha256_file(p) if p.exists() else None,
            }
            for label, p in files.items()
        },
    }
    if extras:
        payload["extras"] = extras
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote manifest -> {out_path}")


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)
