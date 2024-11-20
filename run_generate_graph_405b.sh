#!/bin/bash

#SBATCH --job-name=generate_graph_405b
#SBATCH --output=../outputs/slurm/generate_graph_405b.txt
#SBATCH --partition=long
#SBATCH --mem=20G



# python -u generate_graph.py --style 1 --target ppi --doclevel --simple --model 405b --nebius
# python -u generate_graph.py --style 6 --target tf --doclevel --simple --model 405b --nebius
python -u graphdocs2json.py --style 1 --target ppi --doclevel --simple --model 405b --nebius
python -u graphdocs2json.py --style 6 --target tf --doclevel --simple --model 405b --nebius
