#!/bin/bash

#SBATCH --job-name=cardio_ppi
#SBATCH --output=../../outputs/slurm/cardio_ppi.log
#SBATCH --partition=long
#SBATCH --mem=50G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --target ppi --node g5 --data cardio 
# python -u extract.py --target ppi --node g5 --force_new --data cardio 

