set dotenv-load := true

dreamsim2 q="":
    python backrooms.py --lm "{{q}}" "{{q}}" --template dreamsim2

# Run Backrooms with Hermes x2, Discord + Media (chronicle_fp)
backrooms:
    python backrooms.py --lm hermes hermes --template roleplay --discord chronicle_fp --media chronicle_fp

# Seed dreamsim3 initiator from Supabase (optional query)
# Usage:
#   just seed-dreamsim3                          # random recent dream
#   just seed-dreamsim3 query="rollercoaster"     # fuzzy search
seed-dreamsim3 query="":
    python scripts/seed_dreamsim3.py --query "{{query}}" --print

# Run dreamsim3 with Hermes x2 (no media/discord)
run-dreamsim3:
    python backrooms.py --lm hermes hermes --template dreamsim3

# Seed then run a single dream (random if no query provided)
dreamsim3-one query="" model="gpt5":
    # Print the dream content to the console before the run
    python scripts/seed_dreamsim3.py --query "{{query}}" --print
    python backrooms.py --lm {{model}} {{model}} --template dreamsim3

# Batch: run dreamsim3 across Supabase dreams (default source)
# Usage:
#   just dreamsim3                     # defaults: max=30, models=gpt5,hermes,k2
#   just dreamsim3 50                  # run 50 turns per dream
#   just dreamsim3 30 models="gpt5,k2"  # override models
dreamsim3 max="30" models="gpt5,hermes,k2":
    python scripts/dreamsim3_dataset.py --models "{{models}}" --max-turns {{max}}

# Batch with mixed pairs (model1:model2 entries)
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
