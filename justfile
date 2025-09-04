set dotenv-load := true

# Run Backrooms with Hermes x2, Discord + Media (chronicle_fp)
backrooms:
    python backrooms.py --lm hermes hermes --template roleplay

backrooms2:
    python backrooms.py --lm hermes hermes --discord chronicle_fp --media chronicle_fp
