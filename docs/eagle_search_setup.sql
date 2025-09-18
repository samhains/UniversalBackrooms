-- Enable required extensions (safe to run multiple times)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Optional: a generated tsvector for fast full‑text search
ALTER TABLE eagle_images
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('simple', unaccent(coalesce(title,'') || ' ' || array_to_string(tags,' ')))
  ) STORED;

-- Trigram indexes for fuzzy matching
CREATE INDEX IF NOT EXISTS egi_title_trgm ON eagle_images USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS egi_tags_trgm ON eagle_images USING gin ((array_to_string(tags,' ')) gin_trgm_ops);

-- Full‑text index
CREATE INDEX IF NOT EXISTS egi_tsv_gin ON eagle_images USING gin (tsv);

-- Optional recency helper index
CREATE INDEX IF NOT EXISTS egi_created_at_idx ON eagle_images (created_at DESC);

-- Notes:
-- - tags is expected to be text[]
-- - If you cannot add the generated column, you can still compute the tsvector inline in queries.

