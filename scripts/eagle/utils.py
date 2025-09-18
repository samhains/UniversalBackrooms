"""Shared helpers for working with Eagle image URLs and Supabase storage paths."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def supabase_public_base() -> Optional[str]:
    """Return the base public URL for the eagle-images storage bucket."""
    base = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    if not base:
        return None
    return f"{base}/storage/v1/object/public/eagle-images/"


def storage_path_from_url(url: str) -> Optional[str]:
    """Derive the storage path relative to the eagle-images bucket for a public URL."""
    if not isinstance(url, str) or not url:
        return None
    base = supabase_public_base()
    if not base:
        return None
    if not url.startswith(base):
        return None
    return url[len(base):].lstrip("/") or None


def row_to_image_url(row: Dict[str, Any]) -> Optional[str]:
    """Extract an absolute image URL from a Supabase row if available."""
    # Try explicit URL fields first
    if isinstance(row, dict):
        for key in ("image_url", "imageUrl", "imageurl"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        storage_path = row.get("storage_path") or row.get("path")
        if isinstance(storage_path, str) and storage_path.strip():
            base = supabase_public_base()
            if base:
                cleaned = storage_path.strip().lstrip("/")
                if cleaned:
                    return base + cleaned
    return None
