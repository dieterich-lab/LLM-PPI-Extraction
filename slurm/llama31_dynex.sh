#!/bin/bash

#SBATCH --job-name=dynex
#SBATCH --output=../../outputs/slurm/llama31_dynex.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g3 --port 33 --loglevel off --dynex
python -u extract.py --model llama31regu --node g3 --port 33 --loglevel off --dynex

