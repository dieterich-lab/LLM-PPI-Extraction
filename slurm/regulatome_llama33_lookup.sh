#!/bin/bash

#SBATCH --job-name=ll33_lookup
#SBATCH --output=../../outputs/slurm/regulatome_llama33_lookup.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama33 --node g4 --port 34 --loglevel off  --chattype lookup --extractionmode nerrel --force_cot
python -u extract.py --model llama33regu --node g4 --port 34 --loglevel off  --chattype lookup --extractionmode nerrel --noconfidence

