#!/bin/bash

#SBATCH --job-name=generate_graph
#SBATCH --output=../outputs/slurm/generate_graph.txt
#SBATCH --partition=long
#SBATCH --mem=100G

python -u generate_graph.py --simple --curated --style 2 --model 70b --target both

