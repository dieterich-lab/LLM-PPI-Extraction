#!/bin/bash
#
# LoRA fine-tuning of Llama 3.1 8B on RegulaTome PPI extraction data.
# Runs on a single ampere GPU (gpu-g4-1).
#
# Prerequisites:
#   - .env configured with LINDA_LLM_PYTHON_VENV (defaults to ~/.venvs/finetune)
#   - HF_ACCESS_TOKEN set in environment or .env for model download & upload
#
# Usage:
#   sbatch slurm/finetune_llama31.sh

#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=finetune_ll31
#SBATCH --output=${LINDA_LLM_PROJECT_ROOT:-.}/outputs/slurm/finetune_llama31_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-g4-1
#SBATCH --mem=100G

# ── Load .env configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/../.env" ]]; then
  set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

cd ../ &&
VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/finetune}"
. "$VENV/bin/activate"

python -u finetune.py --model llama31 --noconfidence --train --save --push
