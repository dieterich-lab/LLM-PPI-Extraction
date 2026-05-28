#!/bin/bash

#SBATCH --job-name=names
#SBATCH --output=../../outputs/slurm/regu_names.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u regu_names.py --model llama31 --node g3 --port 33 --loglevel off