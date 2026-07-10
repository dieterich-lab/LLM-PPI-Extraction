#!/bin/bash
#
# LoRA fine-tuning of Llama 3.3 70B on RegulaTome PPI extraction data.
# 2× H100 (hopper) — ~50-60 GB VRAM needed, 80 GB available per GPU.
#
# Prerequisites:
#   - .env configured with LINDA_LLM_PYTHON_VENV (defaults to ~/.venvs/finetune)
#   - HF_ACCESS_TOKEN set in .env for model download & upload
#
# Usage:
#   sbatch slurm/finetune_llama33.sh

#SBATCH --gres=gpu:hopper:2
#SBATCH --job-name=finetune_ll33
#SBATCH --output=../outputs/slurm/finetune_llama33_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-g5-1
#SBATCH --mem=400G

# ── Load .env configuration ────────────────────────────────────────────
if [[ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]]; then
  set -a; source "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/test_linda}"
. "$VENV/bin/activate"

python -u finetune.py --model llama33 --noconfidence --train --save --push

