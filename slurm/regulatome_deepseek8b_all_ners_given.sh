#!/bin/bash

#SBATCH --job-name=ds8b_ner
#SBATCH --output=../../outputs/slurm/regulatome_deepseek_8b_all_ners_given.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate



python -u extract.py --model deepseek8b --extractionmode nerrel --chattype stepwise --data regulatome --target ppi --doclevel docs --node g3 --port 33 --all_ners_given