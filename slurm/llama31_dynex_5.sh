#!/bin/bash

#SBATCH --job-name=31_dynex5
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama31_dynex5.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

# cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g4 --dynex_k 5 
python -u extract.py --model llama31 --node g4 --dynex_k 5 --target tf

