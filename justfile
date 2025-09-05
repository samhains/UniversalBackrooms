set dotenv-load := true

# Run Backrooms with Hermes x2, Discord + Media (chronicle_fp)
backrooms:
    python backrooms.py --lm hermes hermes --template roleplay --discord chronicle_fp --media chronicle_fp

# Seed dreamsim3 initiator from Supabase (optional query)
# Usage:
#   just seed-dreamsim3                  # random recent dream
#   just seed-dreamsim3 q="rollercoaster" # fuzzy search
seed-dreamsim3 q="":
    python scripts/seed_dreamsim3.py --query "{{q}}"

# Run dreamsim3 with Hermes x2 (no media/discord)
run-dreamsim3:
    python backrooms.py --lm hermes hermes --template dreamsim3

# Seed then run (random if no query provided)
dreamsim3 q="":
    python scripts/seed_dreamsim3.py --query "{{q}}"
    python backrooms.py --lm hermes hermes --template dreamsim3
