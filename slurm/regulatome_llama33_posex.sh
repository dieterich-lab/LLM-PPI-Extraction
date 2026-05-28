#!/bin/bash

#SBATCH --job-name=ll33_posex
#SBATCH --output=../../outputs/slurm/regulatome_llama33_poses.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama33 --node g4 --port 34 --loglevel off --examples pos

