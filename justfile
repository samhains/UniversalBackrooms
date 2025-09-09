set dotenv-load := true

# Default: show all tasks
default:
    @just --list

# Current working directory from environment (if needed)
pwd := env_var('PWD')

# DreamSim2: simple two-turn template
dreamsim2 q="":
    python backrooms.py --lm "{{q}}" "{{q}}" --template dreamsim2

# Backrooms with Hermes x2, Discord + Media (chronicle_fp)
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

# Run dreamsim3 with a single model x2 (no media/discord)
run-dreamsim3:
    python backrooms.py --lm hermes hermes --template dreamsim3

# Seed then run a single dream (random if no query provided)
dreamsim3-one query="" model="gpt5":
    # Print the dream content to the console before the run
    python scripts/seed_dreamsim3.py --query "{{query}}" --print
    python backrooms.py --lm {{model}} {{model}} --template dreamsim3

# Batch: run dreamsim3 across Supabase dreams (single count param)
# Usage:
#   just dreamsim3                                  # defaults: max=30, models=gpt5,hermes,k2, source=mine, limit=200
#   just dreamsim3 limit=500                         # fetch and process up to 500 dreams
#   just dreamsim3 max=30 models="gpt5,k2"           # override models
#   just dreamsim3 max=30 source=rsos limit=50000    # many RSOS dreams
dreamsim3 max="30" models="gpt5,hermes,k2" source="mine" limit="200":
    python scripts/dreamsim3_dataset.py --models "{{models}}" --max-turns {{max}} --source {{source}} --limit {{limit}} --max-dreams {{limit}}

# Source-scoped wrappers (limit controls both fetch and processed count)
dreamsim3-mine max="30" models="gpt5,hermes,k2" limit="200":
    python scripts/dreamsim3_dataset.py --source mine --models "{{models}}" --max-turns {{max}} --limit {{limit}} --max-dreams {{limit}}
dreamsim3-rsos max="30" models="gpt5,hermes,k2" limit="200":
    python scripts/dreamsim3_dataset.py --source rsos --models "{{models}}" --max-turns {{max}} --limit {{limit}} --max-dreams {{limit}}
dreamsim3-all max="30" models="gpt5,hermes,k2" limit="200":
    python scripts/dreamsim3_dataset.py --source all --models "{{models}}" --max-turns {{max}} --limit {{limit}} --max-dreams {{limit}}

# Batch with explicit mixed pairs (model1:model2 entries)
#   just dreamsim3-pairs 30 pairs="gpt5:hermes,hermes:k2"
dreamsim3-pairs max="30" pairs="gpt5:hermes,k2:hermes":
    python scripts/dreamsim3_dataset.py --pairs "{{pairs}}" --max-turns {{max}}

# Batch with mixed pairs generated from --models (unique pairs, shuffled)
#   just dreamsim3-mixed 30 models="gpt5,hermes,k2"
dreamsim3-mixed max="30" models="gpt5,hermes,k2":
    python scripts/dreamsim3_dataset.py --models "{{models}}" --mixed --max-turns {{max}}

# Batch with random mixed pairs per dream (N random pairs each)
#   just dreamsim3-mixed-random 30 2 models="gpt5,hermes,k2"
dreamsim3-mixed-random max="30" runs="1" models="gpt5,hermes,k2":
    python scripts/dreamsim3_dataset.py --models "{{models}}" --mixed --mixed-mode random --runs-per-dream {{runs}} --max-turns {{max}}

# Run dreamsim3 over all dreams matching a query (Supabase)
# Usage:
#   just dreamsim3-query query="rollercoaster"          # with defaults
#   just dreamsim3-query  max=20 query="kanye ship"     # override turns
#   just dreamsim3-query  limit=500 models="gpt5,hermes" # control fetch size
dreamsim3-query query="" max="30" models="gpt5,hermes,k2" limit="200":
    python scripts/dreamsim3_dataset.py --query "{{query}}" --limit {{limit}} --models "{{models}}" --max-turns {{max}}
dreamsim3-query-rsos query="" max="30" models="gpt5,hermes,k2" limit="200":
    python scripts/dreamsim3_dataset.py --source rsos --query "{{query}}" --limit {{limit}} --models "{{models}}" --max-turns {{max}}

# DreamSim3 query with Discord + Media (Kie.ai nano banana)
# Usage:
#   just dreamsim3-query-kie query="static"                        # defaults shown below
#   just dreamsim3-query-kie query="rollercoaster" max=30 limit=500 source=all
#   just dreamsim3-query-kie query="monitors" discord="narrative_terminal" media="kieai"
dreamsim3-query-kie query="" max="30" limit="200" source="all" models="sonnet3" discord="narrative_terminal" media="kieai":
    python scripts/dreamsim3_dataset.py \
      --query "{{query}}" \
      --limit {{limit}} \
      --source {{source}} \
      --models "{{models}}" \
      --max-turns {{max}} \
      --discord "{{discord}}" \
      --media "{{media}}"

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
