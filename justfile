# Convenient tasks for UniversalBackrooms

# Default: show help
default:
    @just --list

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

