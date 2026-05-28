#!/bin/bash

#SBATCH --job-name=ll31_new
#SBATCH --output=../../outputs/slurm/llama31_new.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --target ppi --model llama31regu  --node g5
python -u extract.py --target ppi --model llama31regu  --node g5 --examples pos
python -u extract.py --target ppi --model llama31regu  --node g5 --examples neg
python -u extract.py --target ppi --model llama31regu  --node g5 --examples negpos