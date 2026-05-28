#!/bin/bash

#SBATCH --job-name=ll31_papers
#SBATCH --output=../../outputs/slurm/regulatome_llama31_papers.txt
#SBATCH --partition=medium
#SBATCH --mem=10G

cd ../ &&
. ~/.venvs/test_linda/bin/activate

# python -u extract.py --model llama31 --node g3 --loglevel off --data regulatomepapers --examples pos
# Saved json to /beegfs/prj/LINDA_LLM/outputs/triples/regulatomepapers/ppi/llama31/direct/stepwise/docs/pos_ex/triples.json
python -u extract.py --model llama31regu --node g4 --loglevel off --data regulatomepapers --examples pos --port 33
python -u extract.py --model llama31 --node g4 --loglevel off --data regulatomepapers --doclevel chunks  --port 33
python -u extract.py --model llama31regu --node g4 --loglevel off --data regulatomepapers --doclevel chunks  --port 33
python -u extract.py --model llama31 --node g4 --loglevel off --data regulatomepapers --doclevel chunks --examples pos --port 33
python -u extract.py --model llama31regu --node g4 --loglevel off --data regulatomepapers --doclevel chunks --examples pos --port 33

