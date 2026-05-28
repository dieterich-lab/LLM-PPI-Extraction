#!/bin/bash

#SBATCH --job-name=31_tot
#SBATCH --output=/prj/LINDA_LLM/outputs/slurm/llama31_tot.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

# cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g4 --data regulatome  --target ppi --tot 3 --port 37 --tot_strategy vote 
python -u extract.py --model llama31 --node g4 --data regulatome  --target ppi --tot 3 --port 37 --tot_strategy best
python -u extract.py --model llama31 --node g4 --data regulatome  --target ppi --tot 3 --port 37 --tot_strategy merge 

