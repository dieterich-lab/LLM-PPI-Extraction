#!/bin/bash

#SBATCH --gres=gpu:hopper:1
#SBATCH --job-name=finetune
#SBATCH --output=../../outputs/slurm/finetune.txt
#SBATCH --partition=gpu
#SBATCH --mem=300G

cd ../ &&

python -u finetune.py --model llama31 --train --save --target ppi --push

