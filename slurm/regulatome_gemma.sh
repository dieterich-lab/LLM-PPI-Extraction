#!/bin/bash

#SBATCH --job-name=gemma
#SBATCH --output=../../outputs/slurm/regulatome_gemma.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

python -u extract.py --model gemmaregu --node g5 --port 35 --loglevel off
python -u extract.py --model gemmaregu --node g5 --port 35 --loglevel off --examples pos
python -u extract.py --model gemmaregu --node g5 --port 35 --loglevel off --examples neg
python -u extract.py --model gemmaregu --node g5 --port 35 --loglevel off --examples negpos
python -u extract.py --model gemmaregu --node g5 --port 35 --loglevel off --dynex
python -u extract.py --model gemmaregu --node g5 --port 35 --loglevel off --chattype lookup --extractionmode nerrel

