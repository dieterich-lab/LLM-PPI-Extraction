#!/bin/bash

#SBATCH --job-name=ll31_lookup
#SBATCH --output=../../outputs/slurm/regulatome_llama31_lookup.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama31 --node g4 --port 34 --loglevel off  --chattype lookup --extractionmode nerrel
python -u extract.py --model llama31regu --node g4 --port 34 --loglevel off  --chattype lookup --extractionmode nerrel --noconfidence

