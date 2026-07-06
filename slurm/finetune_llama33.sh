#!/bin/bash
#
# LoRA fine-tuning of Llama 3.3 70B on RegulaTome PPI extraction data.
# Prerequisites: set LINDA_LLM_PYTHON_VENV or it defaults to ~/.venvs/finetune

#SBATCH --gres=gpu:hopper:1
#SBATCH --job-name=finetune_ll33
#SBATCH --output=../../outputs/slurm/finetune_llama33.txt
#SBATCH --partition=gpu
#SBATCH --mem=200G

cd ../ &&
VENV="${LINDA_LLM_PYTHON_VENV:-${HOME}/.venvs/finetune}"
. "$VENV/bin/activate"

python -u finetune.py --model llama33 --noconfidence --train --save --push

