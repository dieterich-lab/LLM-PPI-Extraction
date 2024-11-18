#!/bin/bash

#SBATCH --gres=gpu:turing:1
#SBATCH --job-name=marker_parse
#SBATCH --output=../outputs/slurm/marker_parse_lr_eval.txt
#SBATCH --partition=gpu
#SBATCH --mem=50G

. ~/.venvs/linda/bin/activate
python -u marker_parse.py --target lr_eval
