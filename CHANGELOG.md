## Changelog

### 2025-09-12
- scripts/post_lucid_images.py: add `--n` option to request an exact number of images; `--top-k` is deprecated and mapped to `--n`.
- Ensure exactly N images are posted: if the media agent returns fewer than N, top off via Supabase REST with the latest images (deduped).
- Update system prompt dynamically to reflect N (replaces LIMIT 3/THREE with LIMIT N and adds explicit instruction).
- REST fallback limits now respect `--n` rather than a fixed 3.

