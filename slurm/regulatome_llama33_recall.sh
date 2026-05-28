#!/bin/bash

#SBATCH --job-name=ll33_recall
#SBATCH --output=../../outputs/slurm/regulatome_llama33_recall.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model llama33 --node g4 --port 34  --loglevel off --recall
python -u extract.py --model llama33 --node g4 --port 34  --loglevel off --recall --doclevel chunks

