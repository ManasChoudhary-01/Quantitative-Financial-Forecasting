"""Shared helpers for dataset adapters (HTTP download with on-disk cache)."""

from __future__ import annotations

import requests

from config import settings


def download_file(url: str, filename: str, force: bool = False) -> str:
    """
    Download `url` into `data/raw/<filename>` (cached; skipped if present).

    `requests` follows the Hugging Face CDN redirect automatically. Returns the
    local path.
    """
    target = settings.raw_dir / filename
    if target.exists() and not force:
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[download] {filename} already present ({size_mb:.1f} MB)")
        return str(target)

    print(f"[download] Fetching {filename} ...")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    print(f"[download] Saved {filename} -> {target}")
    return str(target)
