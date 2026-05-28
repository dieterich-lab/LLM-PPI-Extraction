#!/bin/bash

#SBATCH --job-name=31_lookup
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama31_lookup.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

# cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g4 --lookup --force_new
python -u extract.py --model llama31 --node g4 --lookup --force_new --target tf 

