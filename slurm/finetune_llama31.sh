#!/bin/bash
#
# LoRA fine-tuning of Llama 3.1 8B on RegulaTome PPI extraction data.
# Runs on a single Turing GPU (gpu-g3-1). VRAM usage ~8 GB, 24 GB sufficient.
#
# Prerequisites:
#   - .env configured with LINDA_LLM_PYTHON_VENV (defaults to ~/.venvs/finetune)
#   - HF_ACCESS_TOKEN set in environment or .env for model download & upload
#
# Usage:
#   sbatch slurm/finetune_llama31.sh

#SBATCH --gres=gpu:turing:1
#SBATCH --job-name=finetune_ll31
#SBATCH --output=../outputs/slurm/finetune_llama31_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-g3-1
#SBATCH --mem=60G

# ── Load .env configuration ────────────────────────────────────────────
if [[ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]]; then
  set -a; source "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/finetune}"
. "$VENV/bin/activate"

python -u finetune.py --model llama31 --noconfidence --train --save --push
