#!/bin/bash

#SBATCH --job-name=ds8b_recall
#SBATCH --output=../../outputs/slurm/regulatome_deepseek8b_recall.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model deepseek8b --recall --node g5 --port 35 --noconfidence --chattype oneshot