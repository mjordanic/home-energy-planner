"""Shared utilities for data fetchers."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import Response
from requests.exceptions import ProxyError

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


def http_get(url: str, timeout: float = 20.0, **kwargs: Any) -> Response:
    """GET with timeout and proxy fallback for hosts blocked by env proxy."""
    logger.debug("http_get: GET %s timeout=%.1f", url, timeout)
    try:
        resp = requests.get(url, timeout=timeout, **kwargs)
        logger.debug("http_get: %s → HTTP %d", url, resp.status_code)
        return resp
    except ProxyError:
        # Some environments inject proxy vars that block certain public endpoints.
        # Retry once with env-derived proxy settings disabled.
        logger.warning("http_get: ProxyError for %s — retrying with trust_env=False", url)
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(url, timeout=timeout, **kwargs)
            logger.debug("http_get (no-proxy): %s → HTTP %d", url, resp.status_code)
            return resp


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
    logger.info("write_manifest: wrote %s", out_path)


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)
