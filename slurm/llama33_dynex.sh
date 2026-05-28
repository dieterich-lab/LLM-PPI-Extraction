#!/bin/bash

#SBATCH --job-name=dynex
#SBATCH --output=../../outputs/slurm/llama33_dynex.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --dynex
python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off --dynex

