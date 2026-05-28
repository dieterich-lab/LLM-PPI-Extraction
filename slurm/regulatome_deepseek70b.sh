#!/bin/bash

#SBATCH --job-name=regu_ds70b
#SBATCH --output=../../outputs/slurm/regulatome_deepseek70b.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model deepseek70b --extractionmode direct --chattype stepwise --data regulatome --target ppi --doclevel docs 
# python -u extract.py --model deepseek70b --extractionmode nerrel --chattype stepwise --data regulatome --target ppi --doclevel docs --all_ners_given