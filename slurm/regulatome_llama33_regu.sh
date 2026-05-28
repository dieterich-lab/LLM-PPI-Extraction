#!/bin/bash

#SBATCH --job-name=ll33_regu
#SBATCH --output=../../outputs/slurm/regulatome_llama33_regu.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama33regu --node g4 --port 34 --noconfidence --loglevel off
python -u extract.py --model llama33regu --node g4 --port 34 --noconfidence --loglevel off --examples negpos
python -u extract.py --model llama33regu --node g4 --port 34 --noconfidence --loglevel off --examples pos
python -u extract.py --model llama33regu --node g4 --port 34 --noconfidence --loglevel off --examples neg
python -u extract.py --model llama33 --node g4 --port 34 --loglevel off

