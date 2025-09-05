set dotenv-load := true

dreamsim2 q="":
    python backrooms.py --lm "{{q}}" "{{q}}" --template dreamsim2

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

# Seed then run a single dream (random if no query provided)
dreamsim3-one q="":
    python scripts/seed_dreamsim3.py --query "{{q}}"
    python backrooms.py --lm hermes hermes --template dreamsim3

# Batch: run dreamsim3 for each CSV dream across models
# Usage:
#   just dreamsim3                     # defaults: max=30, models=gpt5,hermes,k2
#   just dreamsim3 50                  # run 50 turns per dream
#   just dreamsim3 30 models="gpt5,k2"  # override models
dreamsim3 max="30" models="gpt5,hermes,k2" csv="data/dreams_rows.csv":
    python scripts/dreamsim3_dataset.py --csv "{{csv}}" --models "{{models}}" --max-turns {{max}}

# Batch with mixed pairs (model1:model2 entries)
#   just dreamsim3-pairs 30 pairs="gpt5:hermes,hermes:k2"
dreamsim3-pairs max="30" pairs="gpt5:hermes,k2:hermes" csv="data/dreams_rows.csv":
    python scripts/dreamsim3_dataset.py --csv "{{csv}}" --pairs "{{pairs}}" --max-turns {{max}}

# Batch with mixed pairs generated from --models (unique pairs, shuffled)
#   just dreamsim3-mixed 30 models="gpt5,hermes,k2"
dreamsim3-mixed max="30" models="gpt5,hermes,k2" csv="data/dreams_rows.csv":
    python scripts/dreamsim3_dataset.py --csv "{{csv}}" --models "{{models}}" --mixed --max-turns {{max}}
