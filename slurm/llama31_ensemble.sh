#!/bin/bash

#SBATCH --job-name=31_ensemble
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama31_ensemble.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

# cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama33 --node g5 --data regulatome  --target ppi --ensemble 5
python -u extract.py --model llama33 --node g5 --data regulatome  --target tf --ensemble 5

