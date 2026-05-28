#!/bin/bash

#SBATCH --job-name=regu_ds8b
#SBATCH --output=../../outputs/slurm/regulatome_deepseek_8b.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate



python -u extract.py --model deepseek8b --extractionmode direct --chattype stepwise --data regulatome --target ppi --doclevel docs --node g3 --port 33