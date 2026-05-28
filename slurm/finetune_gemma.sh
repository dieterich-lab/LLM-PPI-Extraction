#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=finetune
#SBATCH --output=../../outputs/slurm/finetune_gemma.txt
#SBATCH --partition=gpu
#SBATCH --mem=200G

cd ../ &&
. ~/.venvs/finetune/bin/activate

python -u finetune.py --model gemma --noconfidence --train --save --push

