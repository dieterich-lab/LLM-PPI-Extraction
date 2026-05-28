#!/bin/bash

#SBATCH --job-name=synonyms
#SBATCH --output=../../outputs/slurm/synonyms.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

# python -u synonyms.py --model llama31 --target ppitf --data biored --node g2
python -u synonyms.py --model llama33 --target ppitf --data biored