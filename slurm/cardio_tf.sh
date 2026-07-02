#!/bin/bash

#SBATCH --job-name=cardio_tf
#SBATCH --output=../../outputs/slurm/cardio_tf.txt
#SBATCH --partition=long
#SBATCH --mem=50G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --target tf --node g5 --data cardio
# python -u extract.py --target tf --node g5 --force_new --data cardio

