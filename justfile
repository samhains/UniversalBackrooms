set dotenv-load := true

# Default: show all tasks
default:
    @just --list

# Current working directory from environment (if needed)
pwd := env_var('PWD')

## Quick direct Backrooms (still supported)
dreamsim2 q="":
    python backrooms.py --lm "{{q}}" "{{q}}" --template dreamsim2
backrooms:
    python backrooms.py --lm hermes hermes --template roleplay --discord chronicle_fp --media chronicle_fp

# Seed dreamsim3 initiator from Supabase (optional query)
# Usage:
#   just seed-dreamsim3                          # random recent dream
#   just seed-dreamsim3 query="rollercoaster"     # fuzzy search
seed-dreamsim3 query="":
    python scripts/seed_dreamsim3.py --query "{{query}}" --print
seed-dreamsim3-rsos query="":
    python scripts/seed_dreamsim3.py --source rsos --query "{{query}}" --print

run-dreamsim3:
    python backrooms.py --lm hermes hermes --template dreamsim3

# Seed then run a single dream (random if no query provided)
dreamsim3-one query="" model="gpt5":
    # Print the dream content to the console before the run
    python scripts/seed_dreamsim3.py --query "{{query}}" --print
    python backrooms.py --lm {{model}} {{model}} --template dreamsim3

## DreamSim3 presets via config runner
dreamsim3-default:
    python scripts/run_config.py --config configs/batch_dreamsim3_default.json
dreamsim3-mine:
    python scripts/run_config.py --config configs/batch_dreamsim3_default.json
dreamsim3-rsos:
    python scripts/run_config.py --config configs/batch_dreamsim3_rsos.json
dreamsim3-all:
    python scripts/run_config.py --config configs/batch_dreamsim3_all.json
dreamsim3-pairs:
    python scripts/run_config.py --config configs/batch_dreamsim3_pairs_example.json
dreamsim3-mixed:
    python scripts/run_config.py --config configs/batch_dreamsim3_mixed_all.json
dreamsim3-mixed-random:
    python scripts/run_config.py --config configs/batch_dreamsim3_mixed_random.json
dreamsim3-query:
    python scripts/run_config.py --config configs/batch_dreamsim3_query.json
dreamsim3-query-rsos:
    python scripts/run_config.py --config configs/batch_dreamsim3_query_rsos.json
dreamsim3-query-kie:
    python scripts/run_config.py --config configs/batch_dreamsim3_query_kie.json
dreamsim3-query-kie-nopost:
    python scripts/run_config.py --config configs/batch_dreamsim3_query_kie_nopost.json
dreamsim3-query-multi-media:
    python scripts/run_config.py --config configs/batch_dreamsim3_query_multi_media.json

dreamsim3-query-multi-discord:
    python scripts/run_config.py --config configs/batch_dreamsim3_query_multi_discord.json

# Import RSOS TSV into Supabase (date + content only)
# Usage:
#   just import-rsos                           # imports default file with no limit
#   just import-rsos data/rsos_dream_data.tsv  # explicit file, all rows
#   just import-rsos data/rsos_dream_data.tsv 1000  # explicit limit
import-rsos file="data/rsos_dream_data.tsv" limit="0":
    python scripts/import_rsos_dreams.py --file "{{file}}" --limit {{limit}}

# Convenience wrappers (avoid positional arg confusion)
import-rsos-all:
    python scripts/import_rsos_dreams.py
import-rsos-1000:
    python scripts/import_rsos_dreams.py --limit 1000

# Search dreams in Supabase and print matches (no runs)
# Usage:
#   just dreams-search query="rollercoaster"
#   just dreams-search json=true query="kanye ship" limit=500
dreams-search query="" limit="200" json="false":
    if [ "{{json}}" = "true" ]; then \
      python scripts/search_dreams.py --query "{{query}}" --limit {{limit}} --jsonl; \
    else \
      python scripts/search_dreams.py --query "{{query}}" --limit {{limit}}; \
    fi

# Backfill: upsert backrooms from JSONL + transcripts
sync-backrooms meta="BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl":
    # Cleans tiny/missing logs, rewrites JSONL, then upserts to Supabase
    python scripts/sync_backrooms.py --meta "{{meta}}"

# Prune DB rows for template not present in cleaned metadata
prune-backrooms template="dreamsim3" meta="BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl":
    python scripts/prune_backrooms_db.py --template "{{template}}" --meta "{{meta}}" --delete-not-in-meta

# Export Supabase backrooms to Obsidian folder
# Usage:
#   just obsidian-export                             # full export to ./obsidian
#   just obsidian-export since=2025-09-01            # incremental by date
#   just obsidian-export vault="/path/to/Vault"      # custom vault path
#   just obsidian-export dream_id="<uuid>"           # filter by prompt id
#   just obsidian-export contains="substring"        # filter by prompt text
obsidian-export vault="/Users/samhains/Documents/Backrooms" since="" dream_id="" contains="" limit="1000" index="true" overwrite="true":
    cmd="python scripts/export_obsidian.py --vault \"{{vault}}\" --limit {{limit}}"; \
    if [ -n "{{since}}" ]; then cmd="$cmd --since \"{{since}}\""; fi; \
    if [ -n "{{dream_id}}" ]; then cmd="$cmd --dream-id \"{{dream_id}}\""; fi; \
    if [ -n "{{contains}}" ]; then cmd="$cmd --prompt-contains \"{{contains}}\""; fi; \
    if [ "{{index}}" = "true" ]; then cmd="$cmd --write-index"; fi; \
    if [ "{{overwrite}}" = "true" ]; then cmd="$cmd --overwrite"; fi; \
    eval "$cmd"

# Run from a JSON config (see ./configs for examples)
run config="configs/single_roleplay_hermes.json":
    python scripts/run_config.py --config "{{config}}"

# Sync DreamSim3 and DreamSim4 to Supabase
# Usage:
#   just sync
sync: sync-backrooms sync-dreamsim4

# DreamSim4: cycle through models for N runs (self-dialogue)
# Usage:
#   just dreamsim4                       # uses defaults below
#   just dreamsim4 models="opus4,sonnet4,k2,gpt5,v31" runs=30 turns=30

dreamsim4 models="opus4,sonnet4,k2,gpt5,v31" runs="30" turns="30":
    python scripts/dreamsim4_runner.py \
      --models {{models}} \
      --runs {{runs}} \
      --max-turns {{turns}}

# DreamSim4 with explicit pairs
# Example: just dreamsim4-pairs pairs="opus4:sonnet4,k2:gpt5" turns=30
dreamsim4-pairs pairs turns="30":
    python scripts/dreamsim4_runner.py \
      --pairs {{pairs}} \
      --max-turns {{turns}}

# Sync DreamSim4 runs to Supabase (fixed metadata path)
# Usage: just sync-dreamsim4
sync-dreamsim4:
    python scripts/sync_backrooms.py --meta "BackroomsLogs/dreamsim4/dreamsim4_meta.jsonl"
