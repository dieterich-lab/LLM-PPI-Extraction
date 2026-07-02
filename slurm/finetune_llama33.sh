#!/bin/bash

#SBATCH --gres=gpu:hopper:1
#SBATCH --job-name=finetune_ll33
#SBATCH --output=../../outputs/slurm/finetune_llama33.txt
#SBATCH --partition=gpu
#SBATCH --mem=200G

cd ../ &&
. ~/.venvs/finetune/bin/activate

python -u finetune.py --model llama33 --noconfidence --train --save --push

