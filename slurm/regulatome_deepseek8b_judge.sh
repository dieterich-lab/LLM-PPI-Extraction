#!/bin/bash

#SBATCH --job-name=judge_ds8b
#SBATCH --output=../../outputs/slurm/regulatome_deepseek_8b_judge.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate



python -u judge.py --model deepseek8b --data regulatome --target ppi --doclevel docs --node g3 --port 33 --loglevel off 