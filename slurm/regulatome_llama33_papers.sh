#!/bin/bash

#SBATCH --job-name=ll33_papers
#SBATCH --output=../../outputs/slurm/regulatome_llama33_papers.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate


# python -u extract.py --model llama33 --node g4 --loglevel off --data regulatomepapers --examples pos
# python -u extract.py --model llama33regu --node g4 --loglevel off --data regulatomepapers --examples pos
# python -u extract.py --model llama33 --node g4 --loglevel off --data regulatomepapers --doclevel chunks 
# python -u extract.py --model llama33regu --node g4 --loglevel off --data regulatomepapers --doclevel chunks 
python -u extract.py --model llama33 --node g4 --loglevel off --data regulatomepapers --doclevel chunks --examples pos
python -u extract.py --model llama33regu --node g4 --loglevel off --data regulatomepapers --doclevel chunks --examples pos




