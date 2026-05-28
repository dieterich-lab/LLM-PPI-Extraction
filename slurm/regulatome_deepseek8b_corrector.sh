#!/bin/bash

#SBATCH --job-name=corrector_ds8b
#SBATCH --output=../../outputs/slurm/regulatome_deepseek_8b_corrector.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate



python -u re-evaluate.py --model deepseek8b --data regulatome --target ppi --doclevel docs --node g3 --port 33 --re_evaluate corrector --loglevel off  