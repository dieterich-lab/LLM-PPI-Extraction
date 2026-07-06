#!/bin/bash
#
# Simple single-GPU cardiac PPI extraction via Llama 3.3 70B (Ollama).
# Prerequisites: set LINDA_LLM_PYTHON_VENV or it defaults to ~/.venvs/test_linda

#SBATCH --job-name=cardio_ppi
#SBATCH --output=../../outputs/slurm/cardio_ppi.log
#SBATCH --partition=long
#SBATCH --mem=50G

# ── Load .env configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/../.env" ]]; then
  set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

cd ../ &&
VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/test_linda}"
. "$VENV/bin/activate"

python -u extract.py --target ppi --node g5 --data cardio 
# python -u extract.py --target ppi --node g5 --force_new --data cardio

